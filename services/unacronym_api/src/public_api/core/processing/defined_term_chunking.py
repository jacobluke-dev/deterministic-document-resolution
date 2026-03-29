from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from plainera_unacronym.nlp.extraction.defined_terms.types import (
    TermCandidateScore,
    TermMeaning,
    TermResolution,
    TermResolutionResult,
    TermTier2OccurrenceRanking,
)
from plainera_unacronym.nlp.extraction.tiers.types import Tier2Report

TextSpanTuple = tuple[str, int, int]


def _shift_text_span(span: TextSpanTuple, delta: int) -> TextSpanTuple:
    """Shift a text span by a character offset.

    Args:
        span: Source span as ``(text, start, end)``.
        delta: Character offset to add to ``start`` and ``end``.

    Returns:
        A new shifted span tuple.
    """
    return span[0], int(span[1]) + delta, int(span[2]) + delta


def _shift_optional_text_span(span: TextSpanTuple | None, delta: int) -> TextSpanTuple | None:
    """Shift an optional text span by a character offset.

    Args:
        span: Source span or ``None``.
        delta: Character offset to add to ``start`` and ``end``.

    Returns:
        A shifted span, or ``None`` when the input span is ``None``.
    """
    if span is None:
        return None
    return _shift_text_span(span, delta)


def _shift_term_meaning(meaning: TermMeaning, delta: int) -> TermMeaning:
    return replace(
        meaning,
        intro_span=_shift_text_span(meaning.intro_span, delta),
        definition_span=_shift_optional_text_span(meaning.definition_span, delta),
        alias_target_span=_shift_optional_text_span(meaning.alias_target_span, delta),
    )


def _shift_candidate_score(score: TermCandidateScore, delta: int) -> TermCandidateScore:
    return replace(
        score,
        definition_span=_shift_optional_text_span(score.definition_span, delta),
    )


def _shift_term_resolution(resolution: TermResolution, delta: int) -> TermResolution:
    return replace(
        resolution,
        occurrence_span=_shift_text_span(resolution.occurrence_span, delta),
        chosen_definition_span=_shift_optional_text_span(resolution.chosen_definition_span, delta),
        candidate_scores=tuple(_shift_candidate_score(score, delta) for score in resolution.candidate_scores),
    )


def shift_defined_term_result(
    result: TermResolutionResult,
    delta: int,
) -> TermResolutionResult:
    """Shift all location-bearing spans in a defined-term result.

    This is used when chunk-local results need to be projected back into
    full-document coordinates.

    Args:
        result: Chunk-local term resolution result.
        delta: Character offset to apply to all spans.

    Returns:
        The original result when ``delta == 0``; otherwise a new result with
        shifted spans across meanings, resolutions, candidate scores, and
        undecided entries.
    """
    if delta == 0:
        return result

    shifted_meanings = {
        meaning_id: _shift_term_meaning(meaning, delta)
        for meaning_id, meaning in result.meaning_index.items()
    }

    shifted_term_meaning_index = {
        key: tuple(shifted_meanings[meaning.meaning_id] for meaning in meanings)
        for key, meanings in result.term_meaning_index.items()
    }

    return TermResolutionResult(
        term_meaning_index=shifted_term_meaning_index,
        meaning_index=shifted_meanings,
        term_resolutions=[_shift_term_resolution(res, delta) for res in result.term_resolutions],
        ambiguous_keys=tuple(result.ambiguous_keys),
        undecided=[_shift_term_resolution(res, delta) for res in result.undecided],
        tier2_report=result.tier2_report,
        tier2_ranked=result.tier2_ranked,
    )


def _meaning_identity(meaning: TermMeaning) -> tuple[object, ...]:
    return (
        meaning.normalized_key,
        meaning.surface,
        meaning.intro_span,
        meaning.definition_span,
        meaning.definition_text,
        meaning.intro_kind,
        meaning.section_path,
        meaning.alias_target_span,
        meaning.alias_target_text,
    )


def _meaning_sort_key(meaning: TermMeaning) -> tuple[object, ...]:
    def_start = 10**18 if meaning.definition_span is None else int(meaning.definition_span[1])
    def_end = 10**18 if meaning.definition_span is None else int(meaning.definition_span[2])

    return (
        meaning.normalized_key,
        int(meaning.intro_span[1]),
        int(meaning.intro_span[2]),
        def_start,
        def_end,
        meaning.surface,
        meaning.intro_kind,
    )


def _build_meaning_id(normalized_key: str, ordinal: int) -> str:
    return f"term|{normalized_key}|{ordinal}"


def _resolution_identity(resolution: TermResolution) -> tuple[object, ...]:
    return (
        resolution.occurrence_span,
        resolution.normalized_key,
        resolution.term,
    )


def _resolution_rank(resolution: TermResolution) -> tuple[object, ...]:
    method_rank = {
        "unresolved": 0,
        "tier1": 1,
        "tier2_blend": 2,
    }
    top_score = resolution.candidate_scores[0].total_score if resolution.candidate_scores else 0.0
    resolved = resolution.chosen_meaning_id is not None
    return (
        int(resolved),
        method_rank[resolution.resolution_method],
        float(top_score),
        "" if resolution.chosen_meaning_id is None else resolution.chosen_meaning_id,
    )


def _choose_resolution(current: TermResolution | None, candidate: TermResolution) -> TermResolution:
    if current is None:
        return candidate
    if _resolution_rank(candidate) > _resolution_rank(current):
        return candidate
    return current


def _remap_candidate_score(
    score: TermCandidateScore,
    *,
    old_to_new_meaning_id: dict[str, str],
) -> TermCandidateScore:
    return replace(
        score,
        meaning_id=old_to_new_meaning_id.get(score.meaning_id, score.meaning_id),
    )


def _remap_resolution(
    resolution: TermResolution,
    *,
    old_to_new_meaning_id: dict[str, str],
) -> TermResolution:
    chosen_meaning_id = resolution.chosen_meaning_id
    if chosen_meaning_id is not None:
        chosen_meaning_id = old_to_new_meaning_id.get(chosen_meaning_id, chosen_meaning_id)

    return replace(
        resolution,
        chosen_meaning_id=chosen_meaning_id,
        candidate_scores=tuple(
            _remap_candidate_score(score, old_to_new_meaning_id=old_to_new_meaning_id)
            for score in resolution.candidate_scores
        ),
    )


def _collect_meaning_identity_maps(
    shifted_results: list[TermResolutionResult],
) -> tuple[dict[tuple[object, ...], TermMeaning], list[dict[str, tuple[object, ...]]]]:
    """Collect deduplicated meanings and per-result old-ID identity maps.

    Walks each shifted result and derives a stable identity tuple for every
    meaning using ``_meaning_identity``. The first meaning seen for a given
    identity is retained in the returned deduplicated mapping.

    In parallel, this builds one mapping per input result from the original
    ``meaning_id`` to that meaning identity. These per-result maps are later
    used to remap chunk-local meaning IDs onto rebuilt merged meaning IDs.

    Args:
        shifted_results: Defined-term results whose spans have already been
            shifted into document coordinates.

    Returns:
        A tuple containing:
            - A mapping of meaning identity to the first corresponding
              ``TermMeaning`` encountered across all results.
            - A list aligned to ``shifted_results`` where each item maps the
              original ``meaning_id`` values from that result to meaning
              identity tuples.
    """
    meaning_by_identity: dict[tuple[object, ...], TermMeaning] = {}
    old_id_maps: list[dict[str, tuple[object, ...]]] = []

    for result in shifted_results:
        old_id_to_identity: dict[str, tuple[object, ...]] = {}

        for meaning in result.meaning_index.values():
            identity = _meaning_identity(meaning)
            old_id_to_identity[meaning.meaning_id] = identity
            meaning_by_identity.setdefault(identity, meaning)

        old_id_maps.append(old_id_to_identity)

    return meaning_by_identity, old_id_maps


def _rebuild_meanings(
    meaning_by_identity: dict[tuple[object, ...], TermMeaning],
) -> tuple[list[TermMeaning], dict[tuple[object, ...], str]]:
    ordered_meanings = sorted(meaning_by_identity.values(), key=_meaning_sort_key)

    rebuilt_meanings: list[TermMeaning] = []
    identity_to_new_meaning_id: dict[tuple[object, ...], str] = {}
    ordinal_by_key: dict[str, int] = defaultdict(int)

    for meaning in ordered_meanings:
        ordinal_by_key[meaning.normalized_key] += 1
        ordinal = ordinal_by_key[meaning.normalized_key]
        new_meaning_id = _build_meaning_id(meaning.normalized_key, ordinal)
        rebuilt = replace(
            meaning,
            ordinal=ordinal,
            meaning_id=new_meaning_id,
        )
        rebuilt_meanings.append(rebuilt)
        identity_to_new_meaning_id[_meaning_identity(meaning)] = new_meaning_id

    return rebuilt_meanings, identity_to_new_meaning_id


def _build_term_meaning_index(rebuilt_meanings: list[TermMeaning]) -> dict[str, tuple[TermMeaning, ...]]:
    grouped_meanings: dict[str, list[TermMeaning]] = defaultdict(list)
    for meaning in rebuilt_meanings:
        grouped_meanings[meaning.normalized_key].append(meaning)
    return {
        key: tuple(meanings)
        for key, meanings in grouped_meanings.items()
    }


def _merge_resolution_data(
    shifted_results: list[TermResolutionResult],
    old_id_maps: list[dict[str, tuple[object, ...]]],
    identity_to_new_meaning_id: dict[tuple[object, ...], str],
) -> tuple[
    dict[tuple[object, ...], TermResolution],
    tuple[str, ...],
    Tier2Report | None,
    tuple[TermTier2OccurrenceRanking, ...],
]:
    merged_resolutions: dict[tuple[object, ...], TermResolution] = {}
    ambiguous_keys: set[str] = set()
    tier2_report: Tier2Report | None = None
    tier2_ranked_seen: set[TermTier2OccurrenceRanking] = set()
    tier2_ranked: list[TermTier2OccurrenceRanking] = []

    for result, old_id_to_identity in zip(shifted_results, old_id_maps, strict=True):
        old_id_to_new_meaning_id = {
            old_id: identity_to_new_meaning_id[identity]
            for old_id, identity in old_id_to_identity.items()
            if identity in identity_to_new_meaning_id
        }

        ambiguous_keys.update(result.ambiguous_keys)

        if tier2_report is None and result.tier2_report is not None:
            tier2_report = result.tier2_report

        for item in result.tier2_ranked:
            if item in tier2_ranked_seen:
                continue
            tier2_ranked_seen.add(item)
            tier2_ranked.append(item)

        for resolution in result.term_resolutions:
            remapped = _remap_resolution(
                resolution,
                old_to_new_meaning_id=old_id_to_new_meaning_id,
            )
            key = _resolution_identity(remapped)
            merged_resolutions[key] = _choose_resolution(merged_resolutions.get(key), remapped)

    return merged_resolutions, tuple(sorted(ambiguous_keys)), tier2_report, tuple(tier2_ranked)


def merge_defined_term_results(
    chunk_results: list[tuple[int, TermResolutionResult]],
) -> TermResolutionResult:
    """Merge chunked defined-term results into a single document-level result.

    The merge process:
    1. Shifts chunk-local spans into document coordinates.
    2. Deduplicates meanings by structural identity.
    3. Rebuilds stable meaning IDs and ordinals per normalized key.
    4. Remaps resolution references onto rebuilt meaning IDs.
    5. Chooses the strongest resolution per occurrence identity.

    Args:
        chunk_results: Pairs of ``(delta, result)``, where ``delta`` is the
            chunk start offset in document coordinates.

    Returns:
        A merged, deterministically ordered document-level result.
    """
    shifted_results = [
        shift_defined_term_result(result, delta)
        for delta, result in chunk_results
    ]

    meaning_by_identity, old_id_maps = _collect_meaning_identity_maps(shifted_results)
    rebuilt_meanings, identity_to_new_meaning_id = _rebuild_meanings(meaning_by_identity)

    meaning_index = {meaning.meaning_id: meaning for meaning in rebuilt_meanings}
    term_meaning_index = _build_term_meaning_index(rebuilt_meanings)

    merged_resolutions, ambiguous_keys, tier2_report, tier2_ranked = _merge_resolution_data(
        shifted_results,
        old_id_maps,
        identity_to_new_meaning_id,
    )

    ordered_resolutions = sorted(
        merged_resolutions.values(),
        key=lambda resolution: (
            int(resolution.occurrence_span[1]),
            int(resolution.occurrence_span[2]),
            resolution.normalized_key,
            resolution.term,
        ),
    )

    undecided = [
        resolution
        for resolution in ordered_resolutions
        if resolution.chosen_meaning_id is None
    ]

    return TermResolutionResult(
        term_meaning_index=term_meaning_index,
        meaning_index=meaning_index,
        term_resolutions=ordered_resolutions,
        ambiguous_keys=ambiguous_keys,
        undecided=undecided,
        tier2_report=tier2_report,
        tier2_ranked=tier2_ranked,
    )
