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
        resolver: Any,
        glossary_repo: GlossaryRepository,
        semaphore: Any | None,
        request_timeout_ms: int,
    ) -> None:
        self._resolver = resolver
        self._glossary_repo = glossary_repo
        self._semaphore = semaphore
        self._timeout_s = max(0.001, request_timeout_ms / 1000.0)

    async def resolve(self, payload: ResolveRequest) -> ResolveResponse:
        started = time.perf_counter()

        # Semantic validation: whitespace-only -> 422
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

        # Overload: avoid private attr reads; `locked()` is enough for asyncio.Semaphore semantics.
        if self._semaphore is not None and getattr(self._semaphore, "locked", lambda: False)():
            raise ResolveError(
                http_status=status.HTTP_503_SERVICE_UNAVAILABLE,
                code=ErrorCode.SERVICE_UNAVAILABLE,
                message="Service unavailable.",
                details={"reason": "OVERLOADED"},
            )

        # Detect acronyms by deterministic paren rule; preserve order by first occurrence.
        matches = list(ACRO_PAREN_PATTERN.finditer(payload.text))
        seen: set[str] = set()
        blocks: list[dict[str, Any]] = []

        for m in matches:
            ac = m.group(1)
            if ac in seen:
                continue
            seen.add(ac)

            first_occ = {"start": m.start(1), "end": m.end(1)}

            # Occurrences (optional)
            occs = [{"start": mm.start(1), "end": mm.end(1)} for mm in matches if mm.group(1) == ac]
            occs.sort(key=lambda s: (s["start"], s["end"]))

            # Glossary enrichment (schema: acronyms[].glossary.matches)
            glossary_block: dict[str, Any] | None = None
            if opts.include_glossary_enrichment:
                row = self._glossary_repo.get(acronym=ac)
                if row and (row.get("definition") or ""):
                    glossary_block = {
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

            # Resolver candidates -> acronyms[].definitions
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

            # Deterministic definition ordering
            definitions.sort(key=lambda d: (-float(d["confidence"]), str(d["text"])))

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

        processing_ms = int((time.perf_counter() - started) * 1000)

        return ResolveResponse.model_validate(
            {
                "acronyms": blocks,
                "meta": {
                    "processing_ms": processing_ms,
                    "model_version": _plainera_core_version(),
                    "input_chars": len(payload.text),
                },
            }
        )

    async def _call_resolver(self, acronym: str, *, top_k: int) -> Iterable[DefinitionCandidateLike]:
        from plainera_core.core.domain import Acronym

        async def _invoke() -> Iterable[DefinitionCandidateLike]:
            res = self._resolver.resolve(Acronym(text=acronym), top_k=top_k)
            if inspect.isawaitable(res):
                return await res
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
