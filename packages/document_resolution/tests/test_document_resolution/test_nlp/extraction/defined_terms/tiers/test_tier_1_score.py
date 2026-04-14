from __future__ import annotations

from document_resolution.nlp.detection.defined_terms.types import DefinedTermMention
from document_resolution.nlp.extraction.defined_terms.config import DefinedTermExtractionConfig
from document_resolution.nlp.extraction.defined_terms.tiers.tier_1_score import (
    score_term_occurrence_tier1,
)
from document_resolution.nlp.extraction.defined_terms.types import TermMeaning


def _occ(
    *,
    term: str = "Services",
    start: int = 500,
    end: int = 508,
    normalized_key: str = "services",
    segment_window: str | None = None,
) -> DefinedTermMention:
    return DefinedTermMention(
        term=term,
        start_offset=start,
        end_offset=end,
        normalized_key=normalized_key,
        segment_window=segment_window,
    )


def _meaning(
    *,
    meaning_id: str,
    ordinal: int,
    intro_start: int,
    intro_end: int,
    definition_text: str | None,
    intro_kind: str = "quoted_means",
    section_path: tuple[str, ...] = (),
    surface: str = "Services",
    normalized_key: str = "services",
) -> TermMeaning:
    return TermMeaning(
        meaning_id=meaning_id,
        surface=surface,
        normalized_key=normalized_key,
        ordinal=ordinal,
        intro_span=(surface, intro_start, intro_end),
        definition_span=("definition", intro_end + 1, intro_end + 20) if definition_text else None,
        definition_text=definition_text,
        intro_kind=intro_kind,
        section_path=section_path,
    )


class _StubStructureIndex:
    def __init__(self, mapping: dict[int, tuple[str, ...]]) -> None:
        self._mapping = mapping

    def path_for_offset(self, offset: int) -> tuple[str, ...] | None:
        return self._mapping.get(offset)


class TestScoreTermOccurrenceTier1:
    def test_score_term_occurrence_tier1_returns_empty_ranking_when_no_candidates(self) -> None:
        occ = _occ()
        cfg = DefinedTermExtractionConfig()

        result = score_term_occurrence_tier1(
            text="irrelevant",
            occ=occ,
            candidate_meanings=[],
            structure_index=None,
            cfg=cfg,
        )

        assert result.occ == occ
        assert result.candidate_scores == {}
        assert result.chosen_meaning_id is None
        assert result.gap == 0.0
        assert result.margin == 0.0


    def test_score_term_occurrence_tier1_selects_single_candidate(self) -> None:
        occ = _occ(segment_window="The Services include consultancy services for the customer.")
        meaning = _meaning(
            meaning_id="term|services|1",
            ordinal=1,
            intro_start=100,
            intro_end=108,
            definition_text="consultancy services",
        )
        cfg = DefinedTermExtractionConfig()

        result = score_term_occurrence_tier1(
            text="irrelevant",
            occ=occ,
            candidate_meanings=[meaning],
            structure_index=None,
            cfg=cfg,
        )

        assert result.chosen_meaning_id == "term|services|1"
        assert set(result.candidate_scores) == {"term|services|1"}
        assert result.gap == result.candidate_scores["term|services|1"]
        assert result.margin == result.gap / max(abs(result.gap), 1.0)


    def test_score_term_occurrence_tier1_leaves_near_tie_unresolved_when_prefer_prior_definitions_false(self) -> None:
        occ = _occ(segment_window="The Services are to be provided.")
        earlier = _meaning(
            meaning_id="term|services|1",
            ordinal=1,
            intro_start=100,
            intro_end=108,
            definition_text=None,
        )
        later = _meaning(
            meaning_id="term|services|2",
            ordinal=2,
            intro_start=200,
            intro_end=208,
            definition_text=None,
        )
        cfg = DefinedTermExtractionConfig(
            prefer_prior_definitions=False,
            tier_1_margin_threshold=0.20,
        )

        result = score_term_occurrence_tier1(
            text="irrelevant",
            occ=occ,
            candidate_meanings=[later, earlier],
            structure_index=None,
            cfg=cfg,
        )

        assert result.chosen_meaning_id is None
        assert result.candidate_scores["term|services|1"] == result.candidate_scores["term|services|2"]
        assert result.gap == 0.0
        assert result.margin == 0.0


    def test_score_term_occurrence_tier1_penalises_definitions_introduced_after_occurrence(self) -> None:
        occ = _occ(start=300, end=308, segment_window="The Services are to be provided.")
        prior = _meaning(
            meaning_id="term|services|1",
            ordinal=1,
            intro_start=100,
            intro_end=108,
            definition_text=None,
        )
        future = _meaning(
            meaning_id="term|services|2",
            ordinal=2,
            intro_start=350,
            intro_end=358,
            definition_text=None,
        )
        cfg = DefinedTermExtractionConfig()

        result = score_term_occurrence_tier1(
            text="irrelevant",
            occ=occ,
            candidate_meanings=[future, prior],
            structure_index=None,
            cfg=cfg,
        )

        assert result.candidate_scores["term|services|1"] > result.candidate_scores["term|services|2"]
        assert result.chosen_meaning_id == "term|services|1"


    def test_score_term_occurrence_tier1_uses_section_proximity_to_break_otherwise_similar_candidates(self) -> None:
        occ = _occ(start=500, end=508, segment_window="The Services are to be provided.")
        body = _meaning(
            meaning_id="term|services|1",
            ordinal=1,
            intro_start=100,
            intro_end=108,
            definition_text=None,
            section_path=("body",),
        )
        schedule = _meaning(
            meaning_id="term|services|2",
            ordinal=2,
            intro_start=120,
            intro_end=128,
            definition_text=None,
            section_path=("schedule_a",),
        )
        structure_index = _StubStructureIndex({500: ("schedule_a",)})
        cfg = DefinedTermExtractionConfig()

        result = score_term_occurrence_tier1(
            text="irrelevant",
            occ=occ,
            candidate_meanings=[body, schedule],
            structure_index=structure_index,
            cfg=cfg,
        )

        assert result.candidate_scores["term|services|2"] > result.candidate_scores["term|services|1"]
        assert result.chosen_meaning_id == "term|services|2"
