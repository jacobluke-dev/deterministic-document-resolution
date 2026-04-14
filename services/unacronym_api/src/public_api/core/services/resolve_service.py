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

import time
from importlib import metadata
from typing import Any

from fastapi import status
from plainera_unacronym.orchestration import (
    PIPELINE_ACRONYMS,
    PIPELINE_DEFINED_TERMS,
    PIPELINE_STRUCTURAL_REFERENCES,
    PipelineKey,
    PipelineRegistry,
)
from plainera_unacronym.orchestration.state import OrchestrationState

from public_api.core.errors import ResolveError
from public_api.core.orchestration import Orchestrator
from public_api.core.orchestration.mapper import compose_sections, map_orchestration_state
from public_api.core.orchestration.request_builder import build_orchestration_request
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


def _document_resolution_core_version() -> str:
    """Return a human-readable version string for the Plainera core package.

    The function tries known distribution names in order and falls back to a
    development marker when package metadata is unavailable.

    Returns:
      Version string in the format ``<distribution>@<version>``, or
      ``"document_resolution_core@dev"`` if the installed package version cannot be
      determined.
    """
    for dist_name in ("document_resolution_core", "document_resolution_core"):
        try:
            return f"{dist_name}@{metadata.version(dist_name)}"
        except Exception:
            continue
    return "document_resolution_core@dev"


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

    def _build_response(
        self,
        *,
        text: str,
        started: float,
        resolution_mode: ResolutionMode,
        state: OrchestrationState,
        opts: ResolveOptions,
        lang: str,
    ) -> ResolveResponse:
        """Construct the final public response with timing and input metadata."""
        processing_ms = int((time.perf_counter() - started) * 1000)

        orchestration_meta, errors = map_orchestration_state(state)

        sections = compose_sections(
            state,
            opts=opts,
            lang=lang,
            resolution_mode=resolution_mode,
            glossary_repo=self._glossary_repo,
        )

        return ResolveResponse.model_validate(
            {
                **sections,
                "meta": {
                    "processing_ms": processing_ms,
                    "model_version": _document_resolution_core_version(),
                    "input_chars": len(text),
                    "resolution_mode": resolution_mode,
                },
                "orchestration": orchestration_meta.model_dump(),
                "errors": [e.model_dump() for e in errors],
            }
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
        """Resolve requested pipeline targets and compose the public API response."""
        started = time.perf_counter()
        self._raise_if_overloaded()

        opts, lang = self._validate_and_prepare(payload)
        targets = self._normalise_targets(payload)

        orchestration_request = build_orchestration_request(
            payload,
            targets=targets,
            tier2_model=self._tier2_model,
        )

        orchestrator = Orchestrator(
            pipeline_registry=self._pipeline_registry,
            glossary_repo=self._glossary_repo,
            request_timeout_ms=int(self._timeout_s * 1000),
            tier2_model=self._tier2_model,
        )
        state = await orchestrator.execute_orchestration_request(
            request=orchestration_request,
            opts=opts,
            lang=lang,
            resolution_mode=payload.resolution_mode,
        )

        return self._build_response(
            text=payload.text,
            started=started,
            resolution_mode=payload.resolution_mode,
            state=state,
            opts=opts,
            lang=lang,
        )
