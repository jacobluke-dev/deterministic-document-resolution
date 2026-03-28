from __future__ import annotations

from typing import Any, Protocol

from plainera_unacronym.nlp.common.types import AcronymDetectorResult, ExtractionResult
from plainera_unacronym.nlp.extraction.structural.types import StructuralReferenceResolutionResult

from plainera_unacronym.nlp.extraction.defined_terms.types import (
    TermCandidateScore,
    TermMeaning,
    TermResolutionResult,
)

from public_api.schemas.extraction_types.defined_terms import (
    DefinedTermBlock,
    DefinedTermCandidateBlock,
    DefinedTermMeaningBlock,
)

from public_api.schemas.shared import TextSpan
from public_api.schemas.extraction_types.structural import StructuralReferenceSummaryBlock
from public_api.db.repos.glossary_repo import GlossaryRepository
from public_api.schemas.resolve import ResolveOptions


class _SpanLike(Protocol):
    """Protocol describing a span-like object with integer start and end offsets."""

    start: int
    end: int


def span_start_end(span: Any) -> tuple[int, int]:
    """Extract integer start and end offsets from a span-like value.

    Supports tuple-style spans, objects exposing ``start``/``end``
    attributes, and other two-item iterable values.

    Args:
        span: Span-like value to normalise.

    Returns:
        Tuple of ``(start, end)`` integer offsets.

    Raises:
        TypeError: If the span cannot be interpreted as two integer offsets.
    """
    if isinstance(span, tuple) and len(span) == 2:
        return int(span[0]), int(span[1])

    start = getattr(span, "start", None)
    end = getattr(span, "end", None)
    if isinstance(start, int) and isinstance(end, int):
        return start, end

    try:
        a, b = span
        return int(a), int(b)
    except Exception as exc:
        raise TypeError(f"Unrecognised span type: {type(span)!r} -> {span!r}") from exc


def build_definitions_by_acronym(
    *,
    extr: ExtractionResult,
    opts: ResolveOptions,
) -> dict[str, list[dict[str, Any]]]:
    """Build extracted definition candidates grouped by acronym.

    Picks are added first when they meet the minimum confidence threshold.
    Remaining extracted definitions are then added if they are not exact
    duplicates of an existing candidate for the same acronym. Candidates are
    ordered deterministically with picks first, then by descending confidence,
    then definition text, and are capped by
    ``opts.max_definitions_per_acronym``.

    Args:
        extr: Extraction result containing picks and extracted definitions.
        opts: Resolve options controlling confidence filtering and candidate
            limits.

    Returns:
        Mapping of acronym to ordered extracted definition candidate blocks.
    """
    defs_by_ac: dict[str, list[dict[str, Any]]] = {}

    for key, pick in extr.picks.items():
        if pick is None:
            continue

        conf = float(pick.definition_confidence)
        if conf < float(opts.min_confidence):
            continue

        ds, de = span_start_end(pick.def_span)

        defs_by_ac.setdefault(key, []).append(
            {
                "text": pick.definition,
                "start": ds,
                "end": de,
                "confidence": conf,
                "source": "extracted",
                "_is_pick": True,
            }
        )

    for definition in extr.definitions:
        ac = definition.acronym
        conf = float(definition.definition_confidence)
        if conf < float(opts.min_confidence):
            continue

        cand = {
            "text": definition.definition,
            "start": int(definition.def_start),
            "end": int(definition.def_end),
            "confidence": conf,
            "source": "extracted",
            "_is_pick": False,
        }

        bucket = defs_by_ac.setdefault(ac, [])
        dedupe_key = (cand["text"], cand["start"], cand["end"])
        if not any((x["text"], x["start"], x["end"]) == dedupe_key for x in bucket):
            bucket.append(cand)

    max_k = int(opts.max_definitions_per_acronym)
    for ac, items in defs_by_ac.items():
        items.sort(key=lambda x: (not x["_is_pick"], -float(x["confidence"]), str(x["text"])))
        if max_k > 0:
            defs_by_ac[ac] = items[:max_k]
        for item in defs_by_ac[ac]:
            item.pop("_is_pick", None)

    return defs_by_ac


def maybe_glossary_block(
    *,
    glossary_repo: GlossaryRepository,
    acronym: str,
    lang: str,
    opts: ResolveOptions,
) -> dict[str, Any] | None:
    """Build the glossary enrichment block for one acronym.

    When glossary enrichment is enabled, active glossary meanings with
    non-blank definitions are converted into ``matches`` entries and sorted
    deterministically by domain and definition text.

    Args:
        glossary_repo: Repository used to fetch glossary meanings.
        acronym: Acronym to enrich.
        lang: Language code to attach to glossary matches.
        opts: Resolve options controlling whether glossary enrichment is
            enabled.

    Returns:
        Glossary enrichment block, or ``None`` when enrichment is disabled or
        no usable glossary meanings exist.
    """
    if not opts.include_glossary_enrichment:
        return None

    meanings = glossary_repo.list_meanings(acronym=acronym)
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


def map_acronym_pipeline_to_blocks(
    *,
    det_res: AcronymDetectorResult,
    extr: ExtractionResult,
    opts: ResolveOptions,
    lang: str,
    glossary_repo: GlossaryRepository,
) -> list[dict[str, Any]]:
    """Map detector and extractor outputs into public acronym blocks.

    Occurrences are grouped by acronym, sorted by offset, and used to derive
    each acronym's first occurrence. Extracted definitions are attached from
    ``build_definitions_by_acronym()``, and optional glossary enrichment is
    added via ``maybe_glossary_block()``. Final block ordering is stable by
    first occurrence offset, then acronym text.

    Args:
        det_res: Acronym detector result containing occurrences.
        extr: Extraction result containing definition candidates.
        opts: Resolve options controlling block contents.
        lang: Language code used for glossary enrichment matches.
        glossary_repo: Repository used for optional glossary enrichment.

    Returns:
        Ordered public response blocks for detected acronyms.
    """
    occ_by_ac: dict[str, list[dict[str, int]]] = {}
    first_by_ac: dict[str, dict[str, int]] = {}

    for occurrence in det_res.occurrences:
        ac = occurrence.acronym
        occ_by_ac.setdefault(ac, []).append({"start": occurrence.start_offset, "end": occurrence.end_offset})

    for ac, occs in occ_by_ac.items():
        occs.sort(key=lambda s: (s["start"], s["end"]))
        first_by_ac[ac] = occs[0]

    defs_by_ac = build_definitions_by_acronym(extr=extr, opts=opts)

    acronyms_sorted = sorted(
        first_by_ac.keys(),
        key=lambda acronym: (first_by_ac[acronym]["start"], acronym),
    )

    blocks: list[dict[str, Any]] = []
    for ac in acronyms_sorted:
        first_occ = first_by_ac[ac]
        occs = occ_by_ac.get(ac, [])

        glossary_block = maybe_glossary_block(
            glossary_repo=glossary_repo,
            acronym=ac,
            lang=lang,
            opts=opts,
        )

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


def _map_span(span: tuple[int, int] | None) -> TextSpan | None:
    if span is None:
        return None
    return TextSpan(
        start=int(span[0]),
        end=int(span[1]),
    )

def _map_text_span(span: tuple[str, int, int] | None) -> TextSpan | None:
    if span is None:
        return None
    return TextSpan(
        start=int(span[1]),
        end=int(span[2]),
    )


def map_structural_summary_blocks(
    result: StructuralReferenceResolutionResult,
) -> list[StructuralReferenceSummaryBlock]:
    occurrence_counts: dict[str, int] = {}
    for link in result.links:
        occurrence_counts[link.canonical_key] = occurrence_counts.get(link.canonical_key, 0) + 1

    blocks: list[StructuralReferenceSummaryBlock] = []
    for canonical_key, link in result.unique_links.items():
        blocks.append(
            StructuralReferenceSummaryBlock(
                kind=link.kind,
                canonical_label=link.canonical_label,
                canonical_key=link.canonical_key,
                representative_reference_span=_map_span(link.reference_span),
                representative_target_span=_map_span(link.target_span),
                match_strategy=link.match_strategy,
                strength=float(link.strength),
                provenance=link.provenance,
                resolved=link.target_span is not None and link.match_strategy != "unresolved",
                occurrence_count=occurrence_counts.get(canonical_key, 0),
            )
        )

    blocks.sort(key=lambda b: (b.canonical_key, b.representative_reference_span.start))
    return blocks


def _map_term_meaning(meaning: TermMeaning) -> DefinedTermMeaningBlock:
    return DefinedTermMeaningBlock(
        meaning_id=meaning.meaning_id,
        surface=meaning.surface,
        normalized_key=meaning.normalized_key,
        ordinal=int(meaning.ordinal),
        intro_span=_map_text_span(meaning.intro_span),
        definition_span=_map_text_span(meaning.definition_span),
        definition_text=meaning.definition_text,
        intro_kind=meaning.intro_kind,
        section_path=list(meaning.section_path),
        alias_target_span=_map_text_span(meaning.alias_target_span),
        alias_target_text=meaning.alias_target_text,
    )


def _map_candidate_score(score: TermCandidateScore) -> DefinedTermCandidateBlock:
    return DefinedTermCandidateBlock(
        meaning_id=score.meaning_id,
        total_score=float(score.total_score),
        tier1_score=float(score.tier1_score),
        tier2_score=None if score.tier2_score is None else float(score.tier2_score),
        definition_span=_map_span(score.definition_span),
        components={key: float(value) for key, value in score.components.items()},
    )


def map_defined_term_blocks(result: TermResolutionResult) -> list[DefinedTermBlock]:
    blocks: list[DefinedTermBlock] = []

    for resolution in result.term_resolutions:
        chosen_meaning = None
        if resolution.chosen_meaning_id is not None:
            meaning = result.meaning_index.get(resolution.chosen_meaning_id)
            if meaning is not None:
                chosen_meaning = _map_term_meaning(meaning)

        blocks.append(
            DefinedTermBlock(
                occurrence_span=_map_text_span(resolution.occurrence_span),
                term=resolution.term,
                normalized_key=resolution.normalized_key,
                chosen_meaning_id=resolution.chosen_meaning_id,
                chosen_definition_span=_map_span(resolution.chosen_definition_span),
                resolution_method=resolution.resolution_method,
                resolved=resolution.chosen_meaning_id is not None,
                candidate_scores=[
                    _map_candidate_score(score)
                    for score in resolution.candidate_scores
                ],
                chosen_meaning=chosen_meaning,
            )
        )
    return blocks
