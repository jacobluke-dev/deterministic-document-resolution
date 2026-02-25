# services/unacronym_api/src/public_api/core/services/resolve_service.py
from __future__ import annotations

import asyncio
import inspect
import re
import time
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Iterable

from fastapi import status

from public_api.core.providers import AcronymResolverLike
from public_api.db.repos.glossary_repo import GlossaryRepository
from public_api.schemas.error import ErrorCode
from public_api.schemas.resolve import ResolveOptions, ResolveRequest, ResolveResponse
from public_api.types import DefinitionCandidateLike

ACRO_PAREN_PATTERN = re.compile(r"\(([A-Z][A-Z0-9]{1,9})\)")  # simple + deterministic


def _extract_max_len(model: type[ResolveRequest], field: str) -> int | None:
    """
    Extract `max_length` from Pydantic v2 field metadata (with a harmless v1 fallback).
    Used to produce 413 instead of FastAPI's default 422 for oversize text.
    """
    info = model.model_fields[field]
    for meta in getattr(info, "metadata", []):
        if hasattr(meta, "max_length"):
            return int(meta.max_length)
    return getattr(info, "max_length", None)


TEXT_MAX_LEN = _extract_max_len(ResolveRequest, "text")


def _lang_from_locale(locale: str) -> str:
    # "en-GB" -> "en"
    if not locale:
        return "en"
    return (locale.split("-", 1)[0] or "en").lower()


def _plainera_core_version() -> str:
    for dist_name in ("plainera-core", "plainera_core"):
        try:
            return f"{dist_name}@{metadata.version(dist_name)}"
        except Exception:
            continue
    return "plainera-core@dev"


@dataclass(frozen=True)
class ResolveError(Exception):
    http_status: int
    code: ErrorCode
    message: str
    details: dict[str, Any] | None = None


class ResolveService:
    """
    Domain service for `/v1/resolve`.

    Keeps route handlers thin by encapsulating:
      - acronym detection
      - resolver calls (+ timeout)
      - glossary enrichment
      - deterministic ordering + mapping into the public schema
    """

    def __init__(
        self,
        *,
        resolver: AcronymResolverLike,
        glossary_repo: GlossaryRepository,
        semaphore: Any | None,
        request_timeout_ms: int,
    ) -> None:
        self._resolver = resolver
        self._glossary_repo = glossary_repo
        self._semaphore = semaphore
        self._timeout_s = max(0.001, request_timeout_ms / 1000.0)

    async def resolve(self, payload: ResolveRequest) -> ResolveResponse:
        """
        Resolve acronyms for the given request payload and return the API response model.

        This method is intentionally kept as orchestration-only; the work is delegated to
        helper methods for validation, overload checks, enrichment and mapping.
        """
        started = time.perf_counter()

        opts, lang = self._validate_and_prepare(payload)
        self._raise_if_overloaded()

        matches = list(ACRO_PAREN_PATTERN.finditer(payload.text))
        blocks = await self._build_acronym_blocks(payload.text, matches, opts, lang)

        return self._build_response(payload.text, blocks, started)

    @staticmethod
    def _validate_and_prepare(payload: ResolveRequest) -> tuple[ResolveOptions, str]:
        """Validate semantic constraints and return normalised options + language."""
        # Semantic validation: whitespace-only -> 422
        if not payload.text.strip():
            raise ResolveError(
                http_status=status.HTTP_422_UNPROCESSABLE_ENTITY,
                code=ErrorCode.UNPROCESSABLE_ENTITY,
                message="Text must not be empty.",
                details={"hint": "Provide non-empty 'text'"},
            )

        # Prefer 413 for oversize rather than a FastAPI/Pydantic 422
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
        """Raise 503 if a global concurrency limiter is saturated."""
        if self._semaphore is not None and getattr(self._semaphore, "locked", lambda: False)():
            raise ResolveError(
                http_status=status.HTTP_503_SERVICE_UNAVAILABLE,
                code=ErrorCode.SERVICE_UNAVAILABLE,
                message="Service unavailable.",
                details={"reason": "OVERLOADED"},
            )

    async def _build_acronym_blocks(
        self,
        text: str,
        matches: list[Any],
        opts: ResolveOptions,
        lang: str,
    ) -> list[dict[str, Any]]:
        """Build response blocks for each unique acronym in first-occurrence order."""
        blocks: list[dict[str, Any]] = []
        seen: set[str] = set()

        for ac, first_occ, occs in self.iter_unique_acronyms(matches, seen):
            glossary_block = self._maybe_glossary_block(ac, lang, opts)
            definitions = await self._definitions_for(ac, first_occ, opts)

            block: dict[str, Any] = {
                "acronym": ac,
                "first_occurrence": first_occ,
                "definitions": definitions,
            }
            if opts.return_occurrences:
                block["occurrences"] = occs
            if glossary_block is not None:
                block["glossary"] = glossary_block

            blocks.append(block)

        return blocks

    @staticmethod
    def iter_unique_acronyms(
        matches: list[Any],
        seen: set[str],
    ) -> Iterable[tuple[str, dict[str, int], list[dict[str, int]]]]:
        """Yield (acronym, first_occurrence_span, sorted_occurrences) for each unique acronym."""
        for m in matches:
            ac = m.group(1)
            if ac in seen:
                continue
            seen.add(ac)

            first_occ = {"start": m.start(1), "end": m.end(1)}
            occs = [{"start": mm.start(1), "end": mm.end(1)} for mm in matches if mm.group(1) == ac]
            occs.sort(key=lambda s: (s["start"], s["end"]))

            yield ac, first_occ, occs

    def _maybe_glossary_block(self, ac: str, lang: str, opts: ResolveOptions) -> dict[str, Any] | None:
        """Return the glossary enrichment block if enabled and present."""
        if not opts.include_glossary_enrichment:
            return None

        row = self._glossary_repo.get(acronym=ac)
        if not row or not (row.get("definition") or ""):
            return None

        return {
            "matches": [
                {
                    "definition": row.get("definition") or "",
                    "domain": None,
                    "lang": lang,
                    "confidence": 1.0,
                    "source": "system",
                }
            ]
        }

    async def _definitions_for(
        self,
        ac: str,
        first_occ: dict[str, int],
        opts: ResolveOptions,
    ) -> list[dict[str, Any]]:
        """Call the resolver and map candidates into API definition objects."""
        results = await self._call_resolver(ac, top_k=opts.max_definitions_per_acronym)

        definitions: list[dict[str, Any]] = []
        for c in results:
            score = float(getattr(c, "score", 0.0))
            if score < float(opts.min_confidence):
                continue
            definitions.append(
                {
                    "text": getattr(c, "text", ""),
                    "start": max(0, first_occ["start"] - int(opts.window_chars)),
                    "end": first_occ["end"],
                    "confidence": score,
                    "source": "extracted",
                }
            )

        # Deterministic ordering
        definitions.sort(key=lambda d: (-float(d["confidence"]), str(d["text"])))
        return definitions

    @staticmethod
    def _build_response(
        text: str,
        blocks: list[dict[str, Any]],
        started: float,
    ) -> ResolveResponse:
        """Construct the final ResolveResponse with meta."""
        processing_ms = int((time.perf_counter() - started) * 1000)

        return ResolveResponse.model_validate(
            {
                "acronyms": blocks,
                "meta": {
                    "processing_ms": processing_ms,
                    "model_version": _plainera_core_version(),
                    "input_chars": len(text),
                },
            }
        )

    async def _call_resolver(self, acronym: str, *, top_k: int) -> Iterable[DefinitionCandidateLike]:
        from plainera_core.core.domain import Acronym

        async def _invoke() -> Iterable[DefinitionCandidateLike]:
            res = self._resolver.resolve(Acronym(text=acronym), top_k=top_k)

            if inspect.isawaitable(res):
                out = await res
                return out

            return res

        try:
            if self._semaphore is not None:
                async with self._semaphore:
                    return await asyncio.wait_for(_invoke(), timeout=self._timeout_s)
            return await asyncio.wait_for(_invoke(), timeout=self._timeout_s)
        except asyncio.TimeoutError as exc:
            raise ResolveError(
                http_status=status.HTTP_503_SERVICE_UNAVAILABLE,
                code=ErrorCode.SERVICE_UNAVAILABLE,
                message="Resolution timed out.",
                details={"timeout_ms": int(self._timeout_s * 1000), "acronym": acronym},
            ) from exc
        except ResolveError:
            raise
        except Exception as exc:
            raise ResolveError(
                http_status=status.HTTP_503_SERVICE_UNAVAILABLE,
                code=ErrorCode.SERVICE_UNAVAILABLE,
                message="Resolution failed.",
                details={"acronym": acronym, "reason": str(exc)},
            ) from exc
