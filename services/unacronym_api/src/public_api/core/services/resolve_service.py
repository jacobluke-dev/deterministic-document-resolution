"""
Resolve service orchestration for `/v1/resolve`.

This module exposes three logical layers in the API response:
1) detection/occurrence mapping (`acronym`, `first_occurrence`, `occurrences`);
2) extraction/enrichment evidence (`definitions`, optional `glossary.matches`);
3) deterministic resolution metadata (`candidates`, `selected`, `conflict`, `selection`).

Layer 2 preserves what the pipeline found in the document and what glossary
enrichment returned. Layer 3 does not replace that evidence; it ranks the
available senses deterministically and exposes which candidate was selected
and why.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Iterable, Protocol

import anyio
from fastapi import status
from plainera_unacronym.nlp.common.types import AcronymDetectorResult, ExtractionResult
from plainera_unacronym.nlp.execute import detect_and_extract

from public_api.core.auth.chunking import make_chunks, merge_blocks, shift_blocks
from public_api.core.settings import app_settings
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
        tier2_model: Any | None,
    ) -> None:
        self._glossary_repo = glossary_repo
        self._semaphore = semaphore
        self._timeout_s = max(0.001, request_timeout_ms / 1000.0)
        self._tier2_model = tier2_model

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
        """Return legacy glossary enrichment for an acronym, if enabled.

        This block preserves the older `glossary.matches` response shape for callers
        that still consume curated enrichment separately from UN-75 selection metadata.

        Behaviour:
          - Returns `None` when glossary enrichment is disabled.
          - Fetches all glossary meanings for the acronym via the repository.
          - Includes only active meanings with non-empty definitions.
          - Orders matches deterministically by:
              1) domain (ascending, null/empty first as ""),
              2) definition (ascending),
              3) meaning id via repository ordering.
          - Does not perform sense selection; it simply exposes curated matches.

        Notes:
          - `glossary.matches` is evidence/enrichment, not the final decision layer.
          - UN-75 candidate ranking/selection is added later via
            `_attach_resolution_metadata(...)`.
        """
        if not opts.include_glossary_enrichment:
            return None

        meanings = self._glossary_repo.list_meanings(acronym=ac)
        if not meanings:
            return None

        matches = [
            {
                "definition": str(m.get("definition") or ""),
                "domain": m.get("domain"),
                "lang": lang,
                "confidence": 1.0,
                "source": "system",
            }
            for m in meanings
            if bool(m.get("is_active")) and str(m.get("definition") or "").strip()
        ]

        if not matches:
            return None

        matches.sort(
            key=lambda x: (
                "" if x["domain"] is None else str(x["domain"]).casefold(),
                str(x["definition"]).casefold(),
            )
        )

        return {"matches": matches}

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
        det_res: AcronymDetectorResult,
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
    # TODO unit tests
    def _attach_resolution_metadata(
        self,
        *,
        blocks: list[dict[str, Any]],
        opts: ResolveOptions,
    ) -> list[dict[str, Any]]:
        max_k = int(opts.max_definitions_per_acronym)

        def _norm_definition(text: str) -> str:
            return text.strip().rstrip(" .;,:").casefold()

        def _candidate_sort_key(c: dict[str, Any]) -> tuple[Any, ...]:
            return (
                0 if c.get("provenance") == "document" else 1,
                str(c.get("definition") or "").casefold(),
                "" if c.get("domain") is None else str(c["domain"]).casefold(),
                str(c.get("source_ref") or ""),
            )

        out: list[dict[str, Any]] = []

        for block in blocks:
            nb = dict(block)
            ac = str(nb.get("acronym") or "").strip()
            if not ac:
                out.append(nb)
                continue

            definitions = nb.get("definitions") or []
            meanings = self._glossary_repo.list_meanings(acronym=ac)

            inactive_count = sum(1 for m in meanings if not bool(m.get("is_active")))
            active_meanings = [m for m in meanings if bool(m.get("is_active"))]

            candidates: list[dict[str, Any]] = []

            # 1) document/extracted candidates
            seen_doc_defs: set[str] = set()
            for d in definitions:
                if not isinstance(d, dict):
                    continue

                text = str(d.get("text") or "").strip()
                if not text:
                    continue

                start = d.get("start")
                end = d.get("end")
                source_ref = None
                if isinstance(start, int) and isinstance(end, int):
                    source_ref = f"text_span:{start}-{end}"

                candidates.append(
                    {
                        "domain": None,
                        "definition": text,
                        "score": 0.0,
                        "provenance": "document",
                        "source_ref": source_ref,
                    }
                )
                seen_doc_defs.add(_norm_definition(text))

            # 2) glossary candidates
            for m in active_meanings:
                definition = str(m.get("definition") or "").strip()
                if not definition:
                    continue

                if _norm_definition(definition) in seen_doc_defs:
                    continue

                meaning_id = m.get("meaning_id")
                source_ref = f"sense:{meaning_id}" if meaning_id is not None else None

                candidates.append(
                    {
                        "domain": m.get("domain"),
                        "definition": definition,
                        "score": 0.0,
                        "provenance": "glossary",
                        "source_ref": source_ref,
                    }
                )

            viable_count = len(candidates)

            selected: dict[str, Any] | None = None
            reason: str | None = None

            # 3) deterministic selection
            document_candidates = [c for c in candidates if c.get("provenance") == "document"]
            glossary_candidates = [c for c in candidates if c.get("provenance") == "glossary"]

            if document_candidates:
                document_candidates.sort(key=_candidate_sort_key)
                selected = dict(document_candidates[0])
                reason = "in_document_definition"
            elif len(glossary_candidates) == 1:
                selected = dict(glossary_candidates[0])
                reason = "inactive_filtered" if inactive_count > 0 else "single_candidate"
            elif glossary_candidates:
                glossary_candidates.sort(key=_candidate_sort_key)

                general_candidate = next(
                    (c for c in glossary_candidates if str(c.get("domain") or "").casefold() == "general"),
                    None,
                )
                if general_candidate is not None:
                    selected = dict(general_candidate)
                    reason = "fallback_general"
                else:
                    selected = dict(glossary_candidates[0])
                    reason = "highest_score"

            # 4) selected first, stable ordering, simple MVP scores
            ordered_candidates: list[dict[str, Any]] = []
            if selected is not None:
                for c in candidates:
                    if (
                        c.get("definition") == selected.get("definition")
                        and c.get("domain") == selected.get("domain")
                        and c.get("provenance") == selected.get("provenance")
                        and c.get("source_ref") == selected.get("source_ref")
                    ):
                        c["score"] = 1.0
                    else:
                        c["score"] = 0.0

                remaining = [
                    c
                    for c in candidates
                    if not (
                        c.get("definition") == selected.get("definition")
                        and c.get("domain") == selected.get("domain")
                        and c.get("provenance") == selected.get("provenance")
                        and c.get("source_ref") == selected.get("source_ref")
                    )
                ]
                remaining.sort(key=_candidate_sort_key)

                selected["score"] = 1.0
                ordered_candidates = [selected, *remaining]
            else:
                ordered_candidates = sorted(candidates, key=_candidate_sort_key)

            nb["candidates"] = ordered_candidates[:max_k] if max_k > 0 else []
            nb["selected"] = (
                {
                    "domain": selected.get("domain"),
                    "definition": selected.get("definition"),
                    "reason": reason,
                }
                if selected is not None and reason is not None
                else None
            )
            nb["conflict"] = viable_count > 1
            nb["conflict_count"] = viable_count
            nb["selection"] = {
                "policy_used": None,
                "filtered_inactive_count": inactive_count,
            }
            out.append(nb)
        return out

    async def _run_pipeline(self, text: str, opts: ResolveOptions) -> tuple[AcronymDetectorResult, ExtractionResult]:
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
        text = payload.text

        # Small inputs: unchanged behaviour
        if (not app_settings.CHUNKING_ENABLED) or (len(text) <= app_settings.CHUNK_THRESHOLD_CHARS):
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

            blocks = self._map_pipeline_to_blocks(det_res=det_res, extr=extr, opts=opts, lang=lang)
            blocks = self._attach_resolution_metadata(blocks=blocks, opts=opts)
            return self._build_response(text, blocks, started)

        # Large inputs: chunked mode
        chunks = make_chunks(
            text,
            chunk_size=int(app_settings.CHUNK_SIZE_CHARS),
            overlap=int(app_settings.CHUNK_OVERLAP_CHARS),
        )

        all_blocks: list[list[dict[str, Any]]] = []

        for ch in chunks:
            try:
                det_res, extr = await self._run_pipeline(ch.text, opts)
            except asyncio.TimeoutError as exc:
                raise ResolveError(
                    http_status=status.HTTP_503_SERVICE_UNAVAILABLE,
                    code=ErrorCode.SERVICE_UNAVAILABLE,
                    message="Resolution timed out.",
                    details={"timeout_ms": int(self._timeout_s * 1000), "chunk": {"start": ch.start, "end": ch.end}},
                ) from exc
            except Exception as exc:
                raise ResolveError(
                    http_status=status.HTTP_503_SERVICE_UNAVAILABLE,
                    code=ErrorCode.SERVICE_UNAVAILABLE,
                    message="Resolution failed.",
                    details={"reason": str(exc), "chunk": {"start": ch.start, "end": ch.end}},
                ) from exc

            blocks = self._map_pipeline_to_blocks(det_res=det_res, extr=extr, opts=opts, lang=lang)
            all_blocks.append(shift_blocks(blocks, ch.start))

        merged = merge_blocks(all_blocks)
        merged = self._attach_resolution_metadata(blocks=merged, opts=opts)
        return self._build_response(text, merged, started)
