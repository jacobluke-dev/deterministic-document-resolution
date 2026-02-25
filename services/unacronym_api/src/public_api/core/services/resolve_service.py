from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Iterable, Protocol

import anyio
from fastapi import status
from plainera_unacronym.nlp.common.types import DetectorResult, ExtractionResult
from plainera_unacronym.nlp.execute import detect_and_extract

from public_api.db.repos.glossary_repo import GlossaryRepository
from public_api.schemas.error import ErrorCode
from public_api.schemas.resolve import ResolveOptions, ResolveRequest, ResolveResponse


class _SpanLike(Protocol):
    start: int
    end: int

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
        glossary_repo: GlossaryRepository,
        semaphore: Any | None,
        request_timeout_ms: int,
    ) -> None:
        self._glossary_repo = glossary_repo
        self._semaphore = semaphore
        self._timeout_s = max(0.001, request_timeout_ms / 1000.0)

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

    @staticmethod
    def _span_start_end(span: Any) -> tuple[int, int]:
        # Supports tuple-like spans (start, end) and object spans with attributes.
        if isinstance(span, tuple) and len(span) == 2:
            return int(span[0]), int(span[1])
        start = getattr(span, "start", None)
        end = getattr(span, "end", None)
        if isinstance(start, int) and isinstance(end, int):
            return start, end
        # Fall back to index access for e.g. Span dataclass with __iter__
        try:
            a, b = span
            return int(a), int(b)
        except Exception as exc:
            raise TypeError(f"Unrecognised span type: {type(span)!r} -> {span!r}") from exc

    def _build_definitions_by_acronym(
        self,
        *,
        extr: ExtractionResult,
        opts: ResolveOptions,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Build acronym -> definitions list.

        Ordering rule:
          1) Pick (winner) first, if present.
          2) Remaining candidates from extr.definitions, sorted by (-confidence, text).
        """
        defs_by_ac: dict[str, list[dict[str, Any]]] = {}

        # ---- (1) winners from picks (consumer-facing) ----
        for key, p in extr.picks.items():
            if p is None:
                continue

            conf = float(p.definition_confidence)
            if conf < float(opts.min_confidence):
                continue

            ds, de = self._span_start_end(p.def_span)

            defs_by_ac.setdefault(key, []).append(
                {
                    "text": p.definition,
                    "start": ds,
                    "end": de,
                    "confidence": conf,
                    "source": "extracted",
                    "_is_pick": True,  # internal marker for stable ordering
                }
            )

        # ---- (2) ledger evidence (debug/trace ledger) ----
        for d in extr.definitions:
            ac = d.acronym
            conf = float(d.definition_confidence)
            if conf < float(opts.min_confidence):
                continue

            cand = {
                "text": d.definition,
                "start": int(d.def_start),
                "end": int(d.def_end),
                "confidence": conf,
                "source": "extracted",
                "_is_pick": False,
            }

            bucket = defs_by_ac.setdefault(ac, [])
            dedupe_key = (cand["text"], cand["start"], cand["end"])
            if not any((x["text"], x["start"], x["end"]) == dedupe_key for x in bucket):
                bucket.append(cand)

        # ---- sort + trim, keeping pick first ----
        max_k = int(opts.max_definitions_per_acronym)
        for ac, items in defs_by_ac.items():
            # pick first; rest sorted by confidence desc then text asc
            items.sort(key=lambda x: (not x["_is_pick"], -float(x["confidence"]), str(x["text"])))
            if max_k > 0:
                defs_by_ac[ac] = items[:max_k]
            # strip internal marker
            for x in defs_by_ac[ac]:
                x.pop("_is_pick", None)

        return defs_by_ac

    def _map_pipeline_to_blocks(
        self,
        *,
        det_res: DetectorResult,
        extr: ExtractionResult,
        opts: ResolveOptions,
        lang: str,
    ) -> list[dict[str, Any]]:
        """
        Map pipeline outputs into public API AcronymBlock objects.

        Source of truth:
          - Occurrences come from `det_res.occurrences`.
          - Definition candidates come from `extr.definitions`.
          - Glossary enrichment remains separate under `glossary.matches`.

        Determinism:
          - Acronym blocks ordered by first occurrence (start_offset), tie-break by acronym text.
          - Occurrences ordered by (start, end).
          - Definitions ordered by (-confidence, definition text), trimmed to max_definitions_per_acronym.
        """
        # --- occurrences: surface acronym -> list[Span]
        occ_by_ac: dict[str, list[dict[str, int]]] = {}
        first_by_ac: dict[str, dict[str, int]] = {}

        # Use detector occurrences (surface form preserved)
        for o in det_res.occurrences:
            ac = o.acronym
            occ_by_ac.setdefault(ac, []).append({"start": o.start_offset, "end": o.end_offset})

        # Sort + compute first occurrence
        for ac, occs in occ_by_ac.items():
            occs.sort(key=lambda s: (s["start"], s["end"]))
            first_by_ac[ac] = occs[0]

        # --- winners from picks: surface acronym -> list[Definition]
        defs_by_ac = self._build_definitions_by_acronym(extr=extr, opts=opts)


        # --- build blocks in deterministic order (first occurrence, then acronym)
        acronyms_sorted = sorted(
            first_by_ac.keys(),
            key=lambda acr_sor: (first_by_ac[acr_sor]["start"], acr_sor),
        )

        blocks: list[dict[str, Any]] = []
        for ac in acronyms_sorted:
            first_occ = first_by_ac[ac]
            occs = occ_by_ac.get(ac, [])

            glossary_block = self._maybe_glossary_block(ac, lang, opts)

            block: dict[str, Any] = {
                "acronym": ac,
                "first_occurrence": {"start": first_occ["start"], "end": first_occ["end"]},
                "definitions": defs_by_ac.get(ac, []),
            }

            if opts.return_occurrences:
                block["occurrences"] = occs

            if glossary_block is not None:
                block["glossary"] = glossary_block

            blocks.append(block)

        return blocks

    async def resolve(self, payload: ResolveRequest) -> ResolveResponse:
        """
        Resolve acronyms in the given input text using the full Unacronym pipeline.

        This method is the request-scoped orchestration entrypoint for `/v1/resolve`.
        It performs light request validation and overload checks, runs the core
        detection + extraction pipeline once for the full text, maps the pipeline
        outputs into the public API schema, and returns a `ResolveResponse`.

        Behaviour:
          - Whitespace-only text is rejected in `_validate_and_prepare` (422).
          - Oversize text is rejected in `_validate_and_prepare` (413).
          - If a global semaphore is saturated, `_raise_if_overloaded` raises (503).
          - The pipeline is executed off the event loop in a worker thread.
          - The overall request is bounded by `self._timeout_s`; timeout returns 503
            with a `timeout_ms` detail.
          - Glossary enrichment (if enabled) is applied during mapping, without
            persisting anything to the database (read-only lookup only).

        Args:
            payload: Validated request object containing the raw `text` and optional
                `ResolveOptions` (e.g. `window_chars`, `min_confidence`,
                `include_glossary_enrichment`).

        Returns:
            ResolveResponse: API response containing:
              - `acronyms`: list of `AcronymBlock` objects (occurrences + candidate
                definitions + optional glossary matches)
              - `meta`: processing time and model version metadata

        Raises:
            ResolveError: Wrapped API errors with an HTTP status and `ErrorCode`:
              - 503 SERVICE_UNAVAILABLE when the pipeline times out or fails
                unexpectedly (uniform error body; no internal details leaked).
        """
        started = time.perf_counter()

        opts, lang = self._validate_and_prepare(payload)
        self._raise_if_overloaded()

        # Run the full pipeline once. It’s CPU-ish, so do it off the event loop.
        try:
            det_res, extr = await asyncio.wait_for(
                anyio.to_thread.run_sync(
                    lambda: detect_and_extract(
                        payload.text,
                        det_cfg=None,  # optionally wire real configs here
                        ext_cfg=None,  # optionally wire real configs here
                        window_left=int(opts.window_chars),
                        window_right=int(opts.window_chars),
                        return_reports=False,
                        trace=False,
                        return_state=False,
                    )
                ),
                timeout=self._timeout_s,
            )
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

        blocks = self._map_pipeline_to_blocks(det_res=det_res, extr=extr, opts=opts, lang=lang)

        return self._build_response(payload.text, blocks, started)
