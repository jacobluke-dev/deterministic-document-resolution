"""
Resolve service orchestration for `/v1/resolve`.

This module exposes three logical layers in the API response:
1) detection/occurrence mapping (`acronym`, `first_occurrence`, `occurrences`);
2) extraction/enrichment evidence (`definitions`, optional `glossary.matches`);
3) deterministic resolution metadata (`candidates`, `selected`, `conflict`, `selection`).

Layer 2 preserves what the pipeline found in the document and what glossary
enrichment returned. Layer 3 does not replace that evidence; it ranks the
available meanings deterministically and exposes which candidate was selected
and why.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from importlib import metadata
from typing import Any

import anyio
from fastapi import status
from plainera_unacronym.nlp.common.types import AcronymDetectorResult, ExtractionResult
from plainera_unacronym.nlp.extraction.acronyms.execute import detect_and_extract
from plainera_unacronym.orchestration import (
    PIPELINE_ACRONYMS,
    PIPELINE_DEFINED_TERMS,
    PIPELINE_STRUCTURAL_REFERENCES,
    PipelineKey,
    PipelineRegistry,
)
from plainera_unacronym.orchestration.service import run_selected_pipelines
from plainera_unacronym.orchestration.state import OrchestrationState

from public_api.core.auth.chunking import make_chunks, merge_blocks, shift_blocks
from public_api.core.services.orchestration_mapper import map_orchestration_state
from public_api.core.services.orchestration_request_builder import build_orchestration_request
from public_api.core.services.resolution_policy import attach_resolution_metadata
from public_api.core.services.resolve_mapper import map_pipeline_to_blocks
from public_api.core.settings import app_settings
from public_api.db.repos import GlossaryRepository
from public_api.schemas.error import ErrorCode
from public_api.schemas.resolve import ResolutionMode, ResolveOptions, ResolveRequest, ResolveResponse


def _extract_max_len(model: type[ResolveRequest], field: str) -> int | None:
    """Extract the configured maximum length for a Pydantic model field."""
    info = model.model_fields[field]
    for meta in getattr(info, "metadata", []):
        if hasattr(meta, "max_length"):
            return int(meta.max_length)
    return getattr(info, "max_length", None)


TEXT_MAX_LEN = _extract_max_len(ResolveRequest, "text")


def _lang_from_locale(locale: str) -> str:
    """Normalise a locale string to its base language code.

    Examples:
      - ``"en-GB"`` -> ``"en"``
      - ``"fr-FR"`` -> ``"fr"``

    An empty input falls back to ``"en"``.

    Args:
      locale: Locale string, typically in BCP 47 style such as ``en-GB``.

    Returns:
      Lower-cased base language code.
    """
    if not locale:
        return "en"
    return (locale.split("-", 1)[0] or "en").lower()


def _plainera_core_version() -> str:
    """Return a human-readable version string for the Plainera core package.

    The function tries known distribution names in order and falls back to a
    development marker when package metadata is unavailable.

    Returns:
      Version string in the format ``<distribution>@<version>``, or
      ``"plainera-core@dev"`` if the installed package version cannot be
      determined.
    """
    for dist_name in ("plainera-core", "plainera_core"):
        try:
            return f"{dist_name}@{metadata.version(dist_name)}"
        except Exception:
            continue
    return "plainera-core@dev"


@dataclass(frozen=True)
class ResolveError(Exception):
    """Structured domain exception for resolve endpoint failures.

    Attributes:
      http_status: HTTP status code to return to the caller.
      code: Stable public error code.
      message: Human-readable error message.
      details: Optional structured details for diagnostics and client handling.
    """
    http_status: int
    code: ErrorCode
    message: str
    details: dict[str, Any] | None = None


class ResolveService:
    """Domain service for `/v1/resolve` orchestration."""

    def __init__(
        self,
        *,
        glossary_repo: GlossaryRepository,
        semaphore: Any | None,
        request_timeout_ms: int,
        tier2_model: Any | None,
        pipeline_registry: PipelineRegistry,
    ) -> None:
        """Initialise the resolve service.

        Args:
          glossary_repo: Repository used to fetch glossary meanings for enrichment
            and candidate selection.
          semaphore: Optional global concurrency limiter used to reject work when
            the service is overloaded.
          request_timeout_ms: Maximum request processing time, in milliseconds.
          tier2_model: Optional semantic reranking model passed into the pipeline.
        """
        self._glossary_repo = glossary_repo
        self._semaphore = semaphore
        self._timeout_s = max(0.001, request_timeout_ms / 1000.0)
        self._tier2_model = tier2_model
        self._pipeline_registry = pipeline_registry

    @staticmethod
    def _validate_and_prepare(payload: ResolveRequest) -> tuple[ResolveOptions, str]:
        """Validate request semantics and derive normalised options and language."""
        if not payload.text.strip():
            raise ResolveError(
                http_status=status.HTTP_422_UNPROCESSABLE_ENTITY,
                code=ErrorCode.UNPROCESSABLE_ENTITY,
                message="Text must not be empty.",
                details={"hint": "Provide non-empty 'text'"},
            )

        if TEXT_MAX_LEN is not None and len(payload.text) > int(TEXT_MAX_LEN):
            raise ResolveError(
                http_status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                code=ErrorCode.PAYLOAD_TOO_LARGE,
                message="Body/text too large.",
                details={"limit": int(TEXT_MAX_LEN), "actual": len(payload.text)},
            )

        opts = payload.options or ResolveOptions.model_validate({})
        lang = _lang_from_locale(opts.locale)
        return opts, lang

    def _raise_if_overloaded(self) -> None:
        """Raise a service-unavailable error when the concurrency limiter is saturated."""
        if self._semaphore is not None and getattr(self._semaphore, "locked", lambda: False)():
            raise ResolveError(
                http_status=status.HTTP_503_SERVICE_UNAVAILABLE,
                code=ErrorCode.SERVICE_UNAVAILABLE,
                message="Service unavailable.",
                details={"reason": "OVERLOADED"},
            )

    @staticmethod
    def _build_response(
        text: str,
        blocks: list[dict[str, Any]],
        started: float,
        resolution_mode: str,
        state: OrchestrationState,
    ) -> ResolveResponse:
        """Construct the final public response with timing and input metadata."""
        processing_ms = int((time.perf_counter() - started) * 1000)

        orchestration_meta, errors = map_orchestration_state(state)

        return ResolveResponse.model_validate(
            {
                "acronyms": blocks,
                "meta": {
                    "processing_ms": processing_ms,
                    "model_version": _plainera_core_version(),
                    "input_chars": len(text),
                    "resolution_mode": resolution_mode,
                },
                "orchestration": orchestration_meta.model_dump(),
                "errors": [e.model_dump() for e in errors],
            }
        )

    async def _run_pipeline(self, text: str, opts: ResolveOptions) -> tuple[AcronymDetectorResult, ExtractionResult]:
        """Execute the acronym detection and extraction pipeline with a timeout."""
        return await asyncio.wait_for(
            anyio.to_thread.run_sync(
                lambda: detect_and_extract(
                    text,
                    det_cfg=None,
                    ext_cfg=None,
                    tier2_model=self._tier2_model,
                    window_left=int(opts.window_chars),
                    window_right=int(opts.window_chars),
                    return_reports=False,
                    trace=False,
                    return_state=False,
                )
            ),
            timeout=self._timeout_s,
        )

    async def _run_single_pass(
        self,
        *,
        text: str,
        opts: ResolveOptions,
        lang: str,
        resolution_mode: ResolutionMode,
    ) -> list[dict[str, Any]]:
        """Run the pipeline once and return mapped blocks with resolution metadata."""
        try:
            det_res, extr = await self._run_pipeline(text, opts)
        except asyncio.TimeoutError as exc:
            raise ResolveError(
                http_status=status.HTTP_503_SERVICE_UNAVAILABLE,
                code=ErrorCode.SERVICE_UNAVAILABLE,
                message="Resolution timed out.",
                details={"timeout_ms": int(self._timeout_s * 1000)},
            ) from exc
        except Exception as exc:
            raise ResolveError(
                http_status=status.HTTP_503_SERVICE_UNAVAILABLE,
                code=ErrorCode.SERVICE_UNAVAILABLE,
                message="Resolution failed.",
                details={"reason": str(exc)},
            ) from exc

        blocks = map_pipeline_to_blocks(
            det_res=det_res,
            extr=extr,
            opts=opts,
            lang=lang,
            glossary_repo=self._glossary_repo,
        )
        return attach_resolution_metadata(
            blocks=blocks,
            opts=opts,
            resolution_mode=resolution_mode,
            glossary_repo=self._glossary_repo,
        )

    @staticmethod
    def _normalise_targets(payload: ResolveRequest) -> tuple[PipelineKey, ...]:
        """Return the requested pipeline targets in deterministic execution order.

        If the request does not specify any targets, all supported pipelines are
        returned in the service default order. If targets are provided, duplicate
        entries are removed while preserving the caller's original order.

        Args:
            payload: Resolve request containing the optional target selection.

        Returns:
            Tuple of pipeline keys to execute, with stable ordering and no
            duplicates.
        """
        if payload.targets is None:
            return (
            PIPELINE_ACRONYMS,
            PIPELINE_DEFINED_TERMS,
            PIPELINE_STRUCTURAL_REFERENCES,
            )
        return tuple(dict.fromkeys(target.value for target in payload.targets))

    async def resolve(self, payload: ResolveRequest) -> ResolveResponse:
        """Resolve acronyms in input text using the Unacronym pipeline and API mapping layer."""
        started = time.perf_counter()

        opts, lang = self._validate_and_prepare(payload)
        targets = self._normalise_targets(payload)
        orchestration_request = build_orchestration_request(
            payload,
            targets=targets,
            tier2_model=self._tier2_model,
        )
        state = await run_selected_pipelines(self._pipeline_registry, orchestration_request)
        self._raise_if_overloaded()
        text = payload.text

        if (not app_settings.CHUNKING_ENABLED) or (len(text) <= app_settings.CHUNK_THRESHOLD_CHARS):
            blocks = await self._run_single_pass(
                text=text,
                opts=opts,
                lang=lang,
                resolution_mode=payload.resolution_mode,
            )
            return self._build_response(
                text,
                blocks,
                started,
                payload.resolution_mode,
                state
            )

        chunks = make_chunks(
            text,
            chunk_size=int(app_settings.CHUNK_SIZE_CHARS),
            overlap=int(app_settings.CHUNK_OVERLAP_CHARS),
        )

        all_blocks: list[list[dict[str, Any]]] = []

        for chunk in chunks:
            try:
                det_res, extr = await self._run_pipeline(chunk.text, opts)
            except asyncio.TimeoutError as exc:
                raise ResolveError(
                    http_status=status.HTTP_503_SERVICE_UNAVAILABLE,
                    code=ErrorCode.SERVICE_UNAVAILABLE,
                    message="Resolution timed out.",
                    details={"timeout_ms": int(self._timeout_s * 1000),
                             "chunk": {
                                 "start": chunk.start,
                                 "end": chunk.end}
                             },
                ) from exc
            except Exception as exc:
                raise ResolveError(
                    http_status=status.HTTP_503_SERVICE_UNAVAILABLE,
                    code=ErrorCode.SERVICE_UNAVAILABLE,
                    message="Resolution failed.",
                    details={"reason": str(exc), "chunk": {"start": chunk.start, "end": chunk.end}},
                ) from exc

            blocks = map_pipeline_to_blocks(
                det_res=det_res,
                extr=extr,
                opts=opts,
                lang=lang,
                glossary_repo=self._glossary_repo,
            )
            all_blocks.append(shift_blocks(blocks, chunk.start))

        merged = merge_blocks(all_blocks)
        merged = attach_resolution_metadata(
            blocks=merged,
            opts=opts,
            resolution_mode=payload.resolution_mode,
            glossary_repo=self._glossary_repo,
        )
        return self._build_response(
            text,
            merged,
            started,
            payload.resolution_mode,
            state
        )
