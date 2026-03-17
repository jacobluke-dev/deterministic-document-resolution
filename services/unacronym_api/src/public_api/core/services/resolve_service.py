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
from typing import Any, Protocol

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
    """Protocol describing a span-like object with integer start and end offsets.

    Attributes:
      start: Inclusive start offset.
      end: Exclusive end offset.
    """
    start: int
    end: int

def _extract_max_len(model: type[ResolveRequest], field: str) -> int | None:
    """Extract the configured maximum length for a Pydantic model field.

    This primarily supports Pydantic v2 by inspecting field metadata for a
    ``max_length`` constraint, with a harmless fallback to older attribute-based
    access where available. The value is used to return HTTP 413 for oversized
    input instead of FastAPI's default 422 validation response.

    Args:
      model: Pydantic model class containing the field definition.
      field: Name of the field whose maximum length should be inspected.

    Returns:
      The configured maximum length for the field, or ``None`` if no maximum
      length constraint is defined.
    """
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
    """Domain service for ``/v1/resolve``.

    Keeps route handlers thin by encapsulating request validation, acronym
    detection, pipeline execution, glossary enrichment, deterministic ordering,
    and mapping into the public response schema.
    """

    def __init__(
        self,
        *,
        glossary_repo: GlossaryRepository,
        semaphore: Any | None,
        request_timeout_ms: int,
        tier2_model: Any | None,
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

    @staticmethod
    def _validate_and_prepare(payload: ResolveRequest) -> tuple[ResolveOptions, str]:
        """Validate request semantics and derive normalised options and language.

        This performs semantic checks beyond basic schema validation, including:
          - rejecting whitespace-only text with HTTP 422
          - rejecting oversized text with HTTP 413

        Args:
          payload: Parsed resolve request payload.

        Returns:
          A tuple of:
            - resolved ``ResolveOptions``
            - normalised base language code

        Raises:
          ResolveError: If the input text is empty after trimming or exceeds the
            configured maximum length.
        """
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
        """Raise a service-unavailable error when the concurrency limiter is saturated.

        Raises:
          ResolveError: If a configured semaphore indicates the service is currently
            overloaded.
        """
        if self._semaphore is not None and getattr(self._semaphore, "locked", lambda: False)():
            raise ResolveError(
                http_status=status.HTTP_503_SERVICE_UNAVAILABLE,
                code=ErrorCode.SERVICE_UNAVAILABLE,
                message="Service unavailable.",
                details={"reason": "OVERLOADED"},
            )

    def _maybe_glossary_block(self, ac: str, lang: str, opts: ResolveOptions) -> dict[str, Any] | None:
        """Build the legacy glossary enrichment block for an acronym, if enabled.

        This preserves the older ``glossary.matches`` response shape for clients
        that still consume curated glossary evidence separately from the newer
        candidate selection metadata.

        Args:
          ac: Acronym surface form.
          lang: Normalised response language code.
          opts: Effective resolve options.

        Returns:
          A glossary block containing deterministically ordered matches, or ``None``
          if enrichment is disabled or no valid active meanings exist.
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
        """Construct the final public response with timing and input metadata.

        Args:
          text: Original input text.
          blocks: Mapped acronym blocks to include in the response.
          started: ``time.perf_counter()`` timestamp taken at request start.

        Returns:
          Validated ``ResolveResponse`` containing acronym blocks and meta fields.
        """
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
        """Extract integer start and end offsets from a span-like object.

        Supported forms include:
          - a 2-tuple of ``(start, end)``
          - an object exposing integer ``start`` and ``end`` attributes
          - an iterable that unpacks to two values

        Args:
          span: Span-like value to normalise.

        Returns:
          A ``(start, end)`` tuple of integer offsets.

        Raises:
          TypeError: If the supplied value cannot be interpreted as a span.
        """
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
        """Build a mapping of acronym to ordered extracted definition candidates.

        Ordering rules:
          1. Selected winner from ``extr.picks`` first, if present.
          2. Remaining candidates from ``extr.definitions`` ordered by descending
             confidence and then ascending definition text.

        Duplicate definition spans are removed per acronym, low-confidence
        candidates are filtered out, and results are trimmed according to
        ``max_definitions_per_acronym``.

        Args:
          extr: Extraction pipeline result containing picks and definition ledger
            entries.
          opts: Effective resolve options.

        Returns:
          Mapping of acronym surface form to public definition dictionaries.
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
        """Map detector and extraction outputs into public acronym response blocks.

        Occurrences are sourced from detector output, extracted definitions come
        from the extraction result, and glossary enrichment remains separate under
        ``glossary.matches``.

        Deterministic ordering rules:
          - Acronym blocks ordered by first occurrence, then acronym text
          - Occurrences ordered by ``(start, end)``
          - Definitions ordered by the rules in
            ``_build_definitions_by_acronym(...)``

        Args:
          det_res: Acronym detector result.
          extr: Extraction result including selected picks and candidate ledger.
          opts: Effective resolve options.
          lang: Normalised response language code.

        Returns:
          List of public acronym block dictionaries.
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

    @staticmethod
    def _norm_definition(text: str) -> str:
        """Normalise a definition string for de-duplication and comparison.

        The normalisation trims outer whitespace, removes common trailing terminal
        punctuation, and case-folds the result.

        Args:
          text: Raw definition text.

        Returns:
          Normalised definition string.
        """
        return text.strip().rstrip(" .;,:").casefold()

    @staticmethod
    def _candidate_sort_key(c: dict[str, Any]) -> tuple[Any, ...]:
        """Return a deterministic sort key for resolution candidates.

        Candidates are ordered to prefer document provenance first, then by
        definition text, domain, and source reference.

        Args:
          c: Candidate dictionary.

        Returns:
          Tuple suitable for stable ascending sort order.
        """
        return (
            0 if c.get("provenance") == "document" else 1,
            str(c.get("definition") or "").casefold(),
            "" if c.get("domain") is None else str(c["domain"]).casefold(),
            str(c.get("source_ref") or ""),
        )

    def _build_document_candidates(
        self,
        definitions: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], set[str]]:
        """Build resolution candidates derived from in-document definitions.

        Each valid extracted definition becomes a candidate with ``document``
        provenance. A set of normalised document definitions is also returned for
        later de-duplication against glossary meanings.

        Args:
          definitions: Public definition dictionaries already attached to an
            acronym block.

        Returns:
          A tuple of:
            - list of document-derived candidate dictionaries
            - set of normalised document definition strings
        """
        candidates: list[dict[str, Any]] = []
        seen_doc_defs: set[str] = set()

        for d in definitions:

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
            seen_doc_defs.add(self._norm_definition(text))

        return candidates, seen_doc_defs

    def _build_glossary_candidates(
        self,
        meanings: list[dict[str, Any]],
        seen_doc_defs: set[str],
    ) -> tuple[list[dict[str, Any]], int]:
        """Build glossary-derived resolution candidates excluding document duplicates.

        Inactive meanings are counted for selection metadata but excluded from the
        candidate list. Active meanings with definitions already present in the
        document candidate set are also skipped.

        Args:
          meanings: Glossary meaning records for a given acronym.
          seen_doc_defs: Normalised document definitions already present as
            candidates.

        Returns:
          A tuple of:
            - list of glossary-derived candidate dictionaries
            - count of inactive meanings filtered out
        """
        inactive_count = sum(1 for m in meanings if not bool(m.get("is_active")))
        active_meanings = [m for m in meanings if bool(m.get("is_active"))]

        candidates: list[dict[str, Any]] = []

        for m in active_meanings:
            definition = str(m.get("definition") or "").strip()
            if not definition:
                continue

            if self._norm_definition(definition) in seen_doc_defs:
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

        return candidates, inactive_count

    def _select_resolution_candidate(
        self,
        candidates: list[dict[str, Any]],
        inactive_count: int,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Select the best resolution candidate and explain the selection reason.

        Selection policy:
          1. Prefer document-derived candidates.
          2. If exactly one glossary candidate remains, select it.
          3. Otherwise prefer glossary candidate in the ``general`` domain.
          4. Otherwise fall back to the first deterministically ordered glossary
             candidate.

        Args:
          candidates: All viable document and glossary candidates.
          inactive_count: Number of inactive glossary meanings filtered out earlier.

        Returns:
          A tuple of:
            - selected candidate dictionary, or ``None`` if no candidates exist
            - reason string describing the selection basis, or ``None``
        """
        document_candidates = [c for c in candidates if c.get("provenance") == "document"]
        glossary_candidates = [c for c in candidates if c.get("provenance") == "glossary"]

        if document_candidates:
            document_candidates.sort(key=self._candidate_sort_key)
            return dict(document_candidates[0]), "in_document_definition"

        if len(glossary_candidates) == 1:
            return (
                dict(glossary_candidates[0]),
                "inactive_filtered" if inactive_count > 0 else "single_candidate",
            )

        if glossary_candidates:
            glossary_candidates.sort(key=self._candidate_sort_key)

            general_candidate = next(
                (c for c in glossary_candidates if str(c.get("domain") or "").casefold() == "general"),
                None,
            )
            if general_candidate is not None:
                return dict(general_candidate), "fallback_general"

            return dict(glossary_candidates[0]), "highest_score"

        return None, None

    def _order_candidates(
        self,
        candidates: list[dict[str, Any]],
        selected: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Order candidates deterministically and promote the selected candidate first.

        The selected candidate, when present, is copied and assigned a score of
        ``1.0``. All remaining candidates are assigned ``0.0`` and sorted by the
        standard candidate sort key.

        Args:
          candidates: Unordered viable candidates.
          selected: Candidate chosen by the selection policy, if any.

        Returns:
          Ordered list of candidate dictionaries suitable for the public response.
        """
        if selected is None:
            return sorted(candidates, key=self._candidate_sort_key)

        def _same_candidate(a: dict[str, Any], b: dict[str, Any]) -> bool:
            return (
                a.get("definition") == b.get("definition")
                and a.get("domain") == b.get("domain")
                and a.get("provenance") == b.get("provenance")
                and a.get("source_ref") == b.get("source_ref")
            )

        remaining: list[dict[str, Any]] = []
        for c in candidates:
            if _same_candidate(c, selected):
                c["score"] = 1.0
            else:
                c["score"] = 0.0
                remaining.append(c)

        remaining.sort(key=self._candidate_sort_key)

        selected = dict(selected)
        selected["score"] = 1.0
        return [selected, *remaining]

    def _attach_resolution_metadata(
        self,
        *,
        blocks: list[dict[str, Any]],
        opts: ResolveOptions,
    ) -> list[dict[str, Any]]:
        """Attach candidate, selection, and conflict metadata to acronym blocks.

        For each acronym block, this method:
          - derives document candidates from extracted definitions
          - derives glossary candidates from curated meanings
          - selects a winning candidate using deterministic policy
          - attaches ordered candidates and conflict metadata

        Args:
          blocks: Public acronym blocks before resolution metadata is attached.
          opts: Effective resolve options.

        Returns:
          New acronym blocks enriched with ``candidates``, ``selected``,
          ``conflict``, ``conflict_count``, and ``selection`` fields.
        """
        max_k = int(opts.max_definitions_per_acronym)
        out: list[dict[str, Any]] = []

        for block in blocks:
            nb = dict(block)
            ac = str(nb.get("acronym") or "").strip()
            if not ac:
                out.append(nb)
                continue

            definitions = nb.get("definitions") or []
            meanings = self._glossary_repo.list_meanings(acronym=ac)

            doc_candidates, seen_doc_defs = self._build_document_candidates(definitions)
            glossary_candidates, inactive_count = self._build_glossary_candidates(meanings, seen_doc_defs)

            candidates = [*doc_candidates, *glossary_candidates]
            viable_count = len(candidates)

            selected, reason = self._select_resolution_candidate(candidates, inactive_count)
            ordered_candidates = self._order_candidates(candidates, selected)

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
        """Execute the acronym detection and extraction pipeline with a timeout.

        The pipeline is run in a worker thread so that synchronous core logic does
        not block the async request handler. Execution is bounded by the configured
        request timeout.

        Args:
          text: Source text to analyse.
          opts: Effective resolve options controlling window sizes and thresholds.

        Returns:
          A tuple of:
            - acronym detector result
            - extraction result

        Raises:
          TimeoutError: Propagated by ``asyncio.wait_for`` if pipeline execution
            exceeds the configured timeout.
        """
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
        Resolve acronyms in input text using the Unacronym pipeline and API mapping layer.

        This is the request-scoped orchestration entrypoint for `/v1/resolve`. It validates
        the request, applies overload protection, runs detection/extraction, maps pipeline
        results into public API blocks, and then attaches deterministic resolution metadata.

        The response now exposes three logical layers:
          1) occurrence mapping (`acronym`, `first_occurrence`, `occurrences`);
          2) extraction/enrichment evidence (`definitions`, optional `glossary.matches`);
          3) deterministic resolution metadata (`candidates`, `selected`, `conflict`,
             `conflict_count`, `selection`).

        Behaviour:
          - Whitespace-only text is rejected in `_validate_and_prepare` (422).
          - Oversize text is rejected in `_validate_and_prepare` (413).
          - If a global semaphore is saturated, `_raise_if_overloaded` raises (503).
          - Small inputs are processed in a single pipeline run.
          - Large inputs may be processed in chunked mode, with per-chunk offsets shifted
            back into global coordinates and merged deterministically before resolution
            metadata is attached.
          - The pipeline is executed off the event loop in a worker thread.
          - The overall request is bounded by `self._timeout_s`; timeout returns 503
            with a `timeout_ms` detail.
          - Glossary access is read-only; no server state is mutated.

        Args:
            payload: Validated request containing raw `text` and optional `ResolveOptions`
                such as `window_chars`, `min_confidence`, and
                `include_glossary_enrichment`.

        Returns:
            ResolveResponse: API response containing acronym blocks plus metadata about
            processing time, model version, and input size.

        Raises:
            ResolveError: Wrapped API errors with an HTTP status and `ErrorCode`,
            including:
              - 422 for semantically empty text,
              - 413 for oversized text,
              - 503 for overload, timeout, or pipeline failure.
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
