from __future__ import annotations

import re
from typing import Iterable

from plainera_unacronym.nlp.detection.defined_terms.types import DefinedTermMention
from plainera_unacronym.nlp.extraction.defined_terms.config import DefinedTermExtractionConfig
from plainera_unacronym.nlp.extraction.defined_terms.structure import TermStructureIndex
from plainera_unacronym.nlp.extraction.defined_terms.types import (
    TermSense,
    TermTier1OccurrenceRanking,
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")


def _occurrence_context(
    text: str,
    occ: DefinedTermMention,
    *,
    window_chars: int,
) -> str:
    if occ.segment_window:
        return occ.segment_window

    start = max(0, occ.start_offset - window_chars)
    end = min(len(text), occ.end_offset + window_chars)
    return text[start:end]


def _tokenise(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text)}


def _lexical_overlap_score(
    occ_context: str,
    definition_text: str | None,
    *,
    term_surface: str,
) -> float:
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


def _directionality_score(occ: DefinedTermMention, sense: TermSense) -> float:
    intro_start = sense.intro_span[1]
    return 1.0 if intro_start <= occ.start_offset else -0.25


def _proximity_score(occ: DefinedTermMention, sense: TermSense) -> float:
    intro_end = sense.intro_span[2]
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
    sense: TermSense,
    structure_index: TermStructureIndex | None,
) -> float:
    if structure_index is None:
        return 0.0

    occ_path = structure_index.path_for_offset(occ.start_offset)
    sense_path = sense.section_path

    if occ_path == sense_path:
        return 1.0

    if occ_path and sense_path and occ_path[0] == sense_path[0]:
        return 0.5

    return 0.0


def _intro_kind_score(intro_kind: str) -> float:
    if intro_kind in {"quoted_means", "quoted_shall_mean", "bare_means", "bare_shall_mean"}:
        return 1.0
    if intro_kind == "parenthetical_alias":
        return 0.4
    return 0.0


def _score_candidate(
    *,
    text: str,
    occ: DefinedTermMention,
    sense: TermSense,
    structure_index: TermStructureIndex | None,
    cfg: DefinedTermExtractionConfig,
) -> float:
    occ_context = _occurrence_context(
        text,
        occ,
        window_chars=cfg.tier_1_window_chars,
    )

    directionality = _directionality_score(occ, sense) * cfg.directionality_weight
    proximity = _proximity_score(occ, sense)
    section = _section_proximity_score(occ, sense, structure_index) * cfg.section_proximity_weight
    lexical = (
        _lexical_overlap_score(
            occ_context,
            sense.definition_text,
            term_surface=occ.term,
        )
        * cfg.lexical_overlap_weight
    )
    intro_kind = _intro_kind_score(sense.intro_kind) * cfg.intro_type_weight

    return directionality + proximity + section + lexical + intro_kind


def score_term_occurrence_tier1(
    *,
    text: str,
    occ: DefinedTermMention,
    candidate_senses: Iterable[TermSense],
    structure_index: TermStructureIndex | None,
    cfg: DefinedTermExtractionConfig,
) -> TermTier1OccurrenceRanking:
    senses = list(candidate_senses)

    if not senses:
        return TermTier1OccurrenceRanking(
            occ=occ,
            candidate_scores={},
            chosen_sense_id=None,
            gap=0.0,
            margin=0.0,
        )

    scored = [
        (
            sense,
            _score_candidate(
                text=text,
                occ=occ,
                sense=sense,
                structure_index=structure_index,
                cfg=cfg,
            ),
        )
        for sense in senses
    ]

    scored.sort(key=lambda item: (-item[1], item[0].intro_span[1], item[0].sense_id))
    candidate_scores = {sense.sense_id: score for sense, score in scored}

    top_score = scored[0][1]
    second_score = scored[1][1] if len(scored) > 1 else 0.0
    gap = top_score - second_score
    margin = gap / max(abs(top_score), 1.0)

    chosen_sense_id: str | None = scored[0][0].sense_id
    if len(scored) > 1 and margin < cfg.tier_1_margin_threshold:
        chosen_sense_id = None

    return TermTier1OccurrenceRanking(
        occ=occ,
        candidate_scores=candidate_scores,
        chosen_sense_id=chosen_sense_id,
        gap=gap,
        margin=margin,
    )


def score_term_occurrences_tier1(
    *,
    text: str,
    occurrences: list[DefinedTermMention],
    term_sense_index: dict[str, tuple[TermSense, ...]],
    structure_index: TermStructureIndex | None,
    cfg: DefinedTermExtractionConfig,
) -> list[TermTier1OccurrenceRanking]:
    ranked: list[TermTier1OccurrenceRanking] = []

    for occ in occurrences:
        ranked.append(
            score_term_occurrence_tier1(
                text=text,
                occ=occ,
                candidate_senses=term_sense_index.get(occ.normalized_key, ()),
                structure_index=structure_index,
                cfg=cfg,
            )
        )

    return ranked
