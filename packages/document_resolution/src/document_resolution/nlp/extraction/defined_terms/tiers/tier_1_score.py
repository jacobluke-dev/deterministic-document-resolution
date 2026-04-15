from __future__ import annotations

import re
from collections.abc import Iterable

from document_resolution.nlp.detection.defined_terms.types import DefinedTermMention
from document_resolution.nlp.extraction.defined_terms.config import DefinedTermExtractionConfig
from document_resolution.nlp.extraction.defined_terms.structure import TermStructureIndex
from document_resolution.nlp.extraction.defined_terms.types import (
    TermMeaning,
    TermTier1OccurrenceRanking,
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")


def _occurrence_context(
    text: str,
    occ: DefinedTermMention,
    *,
    window_chars: int,
) -> str:
    """Return local context text for a defined-term occurrence.

    Prefers a precomputed ``segment_window`` on the occurrence when available.
    Otherwise, extracts a symmetric character window around the occurrence span.

    Args:
        text: Full source text containing the occurrence.
        occ: Defined-term occurrence to extract context for.
        window_chars: Number of characters to include on each side of the
            occurrence when no precomputed segment window is present.

    Returns:
        A context string surrounding the occurrence, suitable for deterministic
        Tier-1 feature scoring.
    """
    if occ.segment_window:
        return occ.segment_window

    start = max(0, occ.start_offset - window_chars)
    end = min(len(text), occ.end_offset + window_chars)
    return text[start:end]


def _tokenise(text: str) -> set[str]:
    """Tokenise text into a lowercase word set for overlap scoring.

    Args:
        text: Source text to tokenise.

    Returns:
        A set of lowercase word tokens extracted from ``text``.
    """
    return {m.group(0).lower() for m in _WORD_RE.finditer(text)}


def _lexical_overlap_score(
    occ_context: str,
    definition_text: str | None,
    *,
    term_surface: str,
) -> float:
    """Compute lexical overlap between occurrence context and definition text.

    The surface tokens of the term itself are removed from both sides before
    comparison so the score reflects contextual overlap rather than the repeated
    term string.

    Args:
        occ_context: Local context text around the occurrence.
        definition_text: Definition text for the candidate meaning.
        term_surface: Raw surface form of the occurrence being resolved.

    Returns:
        A normalised overlap score in the range ``[0, 1]``. Returns ``0.0`` when
        the candidate has no definition text or when no informative tokens remain
        after term-token removal.
    """
    if not definition_text:
        return 0.0

    occ_tokens = _tokenise(occ_context)
    def_tokens = _tokenise(definition_text)
    term_tokens = _tokenise(term_surface)

    occ_tokens -= term_tokens
    def_tokens -= term_tokens

    if not occ_tokens or not def_tokens:
        return 0.0

    overlap = occ_tokens & def_tokens
    return len(overlap) / len(def_tokens)


def _directionality_score(occ: DefinedTermMention, meaning: TermMeaning) -> float:
    """Score whether the candidate introduction appears before the occurrence.

    Earlier introductions are preferred. Candidate meanings introduced after the
    occurrence receive a small penalty.

    Args:
        occ: Defined-term occurrence being resolved.
        meaning: Candidate meaning being scored.

    Returns:
        A directionality score favouring prior introductions.
    """
    intro_start = meaning.intro_span[1]
    return 1.0 if intro_start <= occ.start_offset else -0.25


def _proximity_score(occ: DefinedTermMention, meaning: TermMeaning) -> float:
    """Score proximity between the occurrence and candidate introduction span.

    Nearby introductions receive a higher score than distant ones using a small
    bucketed distance heuristic.

    Args:
        occ: Defined-term occurrence being resolved.
        meaning: Candidate meaning being scored.

    Returns:
        A proximity score favouring nearby introductions.
    """
    intro_end = meaning.intro_span[2]
    dist = abs(occ.start_offset - intro_end)

    if dist <= 250:
        return 1.0
    if dist <= 1000:
        return 0.7
    if dist <= 5000:
        return 0.35
    return 0.1


def _section_proximity_score(
    occ: DefinedTermMention,
    meaning: TermMeaning,
    structure_index: TermStructureIndex | None,
) -> float:
    """Score structural proximity between an occurrence and a candidate meaning.

    Exact section-path matches score highest. Candidates sharing only the top-level
    section receive a partial score. When no structure index is available, the
    score is ``0.0``.

    Args:
        occ: Defined-term occurrence being resolved.
        meaning: Candidate meaning being scored.
        structure_index: Optional structure index used to map offsets to section
            paths.

    Returns:
        A structural proximity score in the range ``[0.0, 1.0]``.
    """
    if structure_index is None:
        return 0.0

    occ_path = structure_index.path_for_offset(occ.start_offset)
    meaning_path = meaning.section_path

    if occ_path == meaning_path:
        return 1.0

    if occ_path and meaning_path and occ_path[0] == meaning_path[0]:
        return 0.5

    return 0.0


def _intro_kind_score(intro_kind: str) -> float:
    """Return a heuristic weight for the introduction style of a candidate meaning.

    More explicit definitional patterns such as ``quoted_means`` and
    ``quoted_shall_mean`` receive a stronger score than weaker patterns such as a
    parenthetical alias.

    Args:
        intro_kind: Introduction kind label recorded on the candidate meaning.

    Returns:
        A heuristic intro-kind score used as one Tier-1 scoring component.
    """
    if intro_kind in {"quoted_means", "quoted_shall_mean", "bare_means", "bare_shall_mean"}:
        return 1.0
    if intro_kind == "parenthetical_alias":
        return 0.4
    return 0.0


def _score_candidate(
    *,
    text: str,
    occ: DefinedTermMention,
    meaning: TermMeaning,
    structure_index: TermStructureIndex | None,
    cfg: DefinedTermExtractionConfig,
) -> float:
    """Compute the deterministic Tier-1 score for one occurrence/meaning pair.

    The final score is a weighted sum of directional, proximity, structural,
    lexical-overlap, and introduction-kind components.

    Args:
        text: Full source text containing the occurrence.
        occ: Defined-term occurrence being resolved.
        meaning: Candidate meaning to score.
        structure_index: Optional structure index used for section-path proximity.
        cfg: Active extraction configuration controlling Tier-1 weights and
            context-window size.

    Returns:
        The final deterministic Tier-1 score for the candidate meaning.
    """
    occ_context = _occurrence_context(
        text,
        occ,
        window_chars=cfg.tier_1_window_chars,
    )

    directionality = _directionality_score(occ, meaning) * cfg.directionality_weight
    proximity = _proximity_score(occ, meaning)
    section = _section_proximity_score(occ, meaning, structure_index) * cfg.section_proximity_weight
    lexical = (
        _lexical_overlap_score(
            occ_context,
            meaning.definition_text,
            term_surface=occ.term,
        )
        * cfg.lexical_overlap_weight
    )
    intro_kind = _intro_kind_score(meaning.intro_kind) * cfg.intro_type_weight

    return directionality + proximity + section + lexical + intro_kind


def score_term_occurrence_tier1(
    *,
    text: str,
    occ: DefinedTermMention,
    candidate_meanings: Iterable[TermMeaning],
    structure_index: TermStructureIndex | None,
    cfg: DefinedTermExtractionConfig,
) -> TermTier1OccurrenceRanking:
    """Score one defined-term occurrence against its candidate meanings.

    Candidates are scored deterministically and sorted by descending score.

    Selection semantics:
    - If there is a clear winner above the configured margin threshold, select it.
    - If the top candidates are not separable within the configured margin:
      - when ``prefer_prior_definitions`` is enabled, select the earliest
        introduced candidate by document order
      - otherwise, leave the occurrence unresolved

    Args:
        text: Full source text containing the occurrence.
        occ: Defined-term occurrence to resolve.
        candidate_meanings: Candidate meanings associated with the occurrence's
            normalised term key.
        structure_index: Optional structure index used for section-path scoring.
        cfg: Active extraction configuration controlling Tier-1 thresholds and
            feature weights.

    Returns:
        A ``TermTier1OccurrenceRanking`` containing candidate scores, the chosen
        meaning ID when deterministically selectable, and the computed gap and
        margin between the top two candidates.
    """
    meanings = list(candidate_meanings)

    if not meanings:
        return TermTier1OccurrenceRanking(
            occ=occ,
            candidate_scores={},
            chosen_meaning_id=None,
            gap=0.0,
            margin=0.0,
        )

    scored = [
        (
            meaning,
            _score_candidate(
                text=text,
                occ=occ,
                meaning=meaning,
                structure_index=structure_index,
                cfg=cfg,
            ),
        )
        for meaning in meanings
    ]

    scored.sort(key=lambda item: (-item[1], item[0].intro_span[1], item[0].meaning_id))
    candidate_scores = {meaning.meaning_id: score for meaning, score in scored}

    top_score = scored[0][1]
    second_score = scored[1][1] if len(scored) > 1 else 0.0
    gap = top_score - second_score
    scale = max(abs(top_score), 1.0)
    margin = gap / scale

    chosen_meaning_id: str | None = scored[0][0].meaning_id

    if len(scored) > 1 and margin < cfg.tier_1_margin_threshold:
        if cfg.prefer_prior_definitions:
            candidates_within_margin = [
                meaning for meaning, score in scored if (top_score - score) / scale <= cfg.tier_1_margin_threshold
            ]
            chosen_meaning_id = min(
                candidates_within_margin,
                key=lambda meaning: (meaning.intro_span[1], meaning.meaning_id),
            ).meaning_id
        else:
            chosen_meaning_id = None

    return TermTier1OccurrenceRanking(
        occ=occ,
        candidate_scores=candidate_scores,
        chosen_meaning_id=chosen_meaning_id,
        gap=gap,
        margin=margin,
    )


def score_term_occurrences_tier1(
    *,
    text: str,
    occurrences: list[DefinedTermMention],
    term_meaning_index: dict[str, tuple[TermMeaning, ...]],
    structure_index: TermStructureIndex | None,
    cfg: DefinedTermExtractionConfig,
) -> list[TermTier1OccurrenceRanking]:
    """Run deterministic Tier-1 scoring for all detected term occurrences.

    Each occurrence is matched against the candidate meanings for its normalised
    term key and scored independently using ``score_term_occurrence_tier1``.

    Args:
        text: Full source text containing the occurrences.
        occurrences: Detected later references to defined terms.
        term_meaning_index: Mapping from normalised term key to candidate meanings.
        structure_index: Optional structure index used for section-path scoring.
        cfg: Active extraction configuration controlling Tier-1 thresholds and
            feature weights.

    Returns:
        A list of ``TermTier1OccurrenceRanking`` objects aligned to the input
        ``occurrences`` order.
    """
    ranked: list[TermTier1OccurrenceRanking] = []

    for occ in occurrences:
        ranked.append(
            score_term_occurrence_tier1(
                text=text,
                occ=occ,
                candidate_meanings=term_meaning_index.get(occ.normalized_key, ()),
                structure_index=structure_index,
                cfg=cfg,
            )
        )

    return ranked
