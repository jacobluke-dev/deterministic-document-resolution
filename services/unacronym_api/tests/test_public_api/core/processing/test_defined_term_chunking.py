import inspect

from plainera_unacronym.nlp.extraction.defined_terms.types import (
    TermCandidateScore,
    TermMeaning,
    TermResolution,
    TermResolutionResult,
)
from public_api.core.processing.defined_term_chunking import (
    merge_defined_term_results,
    shift_defined_term_result,
)


def _make_candidate_score(
    *,
    meaning_id: str,
    total_score: float,
    definition_span: tuple[str, int, int] | None,
) -> TermCandidateScore:
    parameters = inspect.signature(TermCandidateScore).parameters

    candidates = {
        "meaning_id": meaning_id,
        "definition_span": definition_span,
        "total_score": total_score,
        "score": total_score,
        "blended_score": total_score,
        "rank_score": total_score,
        "candidate_score": total_score,
        "tier1_score": total_score,
        "tier2_score": 0.0,
        "lexical_score": total_score,
        "structural_score": 0.0,
        "section_score": 0.0,
        "occurrence_score": 0.0,
        "components": {},
    }

    kwargs = {
        name: value
        for name, value in candidates.items()
        if name in parameters
    }

    missing_required = [
        name
        for name, parameter in parameters.items()
        if parameter.default is inspect._empty and name not in kwargs
    ]
    if missing_required:
        raise AssertionError(
            f"Update _make_candidate_score for required TermCandidateScore fields: {missing_required}"
        )

    return TermCandidateScore(**kwargs)


class TestShiftDefinedTermResult:
    def test_shifts_all_spans_in_meanings_resolutions_and_undecided(self) -> None:
        meaning = TermMeaning(
            meaning_id="term|services|1",
            surface="Services",
            normalized_key="services",
            ordinal=1,
            intro_span=("Services", 10, 18),
            definition_span=("consulting services", 25, 43),
            definition_text="consulting services",
            intro_kind="means",
            section_path=("1",),
            alias_target_span=("services", 50, 58),
            alias_target_text="services",
        )
        score = _make_candidate_score(
            meaning_id="term|services|1",
            total_score=0.7,
            definition_span=("consulting services", 25, 43),
        )
        resolution = TermResolution(
            occurrence_span=("Services", 100, 108),
            term="Services",
            normalized_key="services",
            chosen_meaning_id="term|services|1",
            chosen_definition_span=("consulting services", 25, 43),
            candidate_scores=(score,),
            resolution_method="tier1",
        )
        undecided = TermResolution(
            occurrence_span=("Services", 150, 158),
            term="Services",
            normalized_key="services",
            chosen_meaning_id=None,
            chosen_definition_span=None,
            candidate_scores=(),
            resolution_method="unresolved",
        )
        result = TermResolutionResult(
            term_meaning_index={"services": (meaning,)},
            meaning_index={meaning.meaning_id: meaning},
            term_resolutions=[resolution],
            ambiguous_keys=("services",),
            undecided=[undecided],
            tier2_report=None,
            tier2_ranked=(),
        )

        shifted = shift_defined_term_result(result, 200)

        shifted_meaning = shifted.meaning_index["term|services|1"]
        assert shifted_meaning.intro_span == ("Services", 210, 218)
        assert shifted_meaning.definition_span == ("consulting services", 225, 243)
        assert shifted_meaning.alias_target_span == ("services", 250, 258)

        shifted_resolution = shifted.term_resolutions[0]
        assert shifted_resolution.occurrence_span == ("Services", 300, 308)
        assert shifted_resolution.chosen_definition_span == ("consulting services", 225, 243)
        assert shifted_resolution.candidate_scores[0].definition_span == ("consulting services", 225, 243)

        shifted_undecided = shifted.undecided[0]
        assert shifted_undecided.occurrence_span == ("Services", 350, 358)

    def test_returns_original_object_when_delta_is_zero(self) -> None:
        result = TermResolutionResult(
            term_meaning_index={},
            meaning_index={},
            term_resolutions=[],
            ambiguous_keys=(),
            undecided=[],
            tier2_report=None,
            tier2_ranked=(),
        )

        shifted = shift_defined_term_result(result, 0)

        assert shifted is result


class TestMergeDefinedTermResults:
    def test_merges_duplicate_meanings_and_prefers_stronger_resolution(self) -> None:
        meaning_chunk_1 = TermMeaning(
            meaning_id="chunk1-meaning",
            surface="Services",
            normalized_key="services",
            ordinal=99,
            intro_span=("Services", 10, 18),
            definition_span=("consulting services", 25, 43),
            definition_text="consulting services",
            intro_kind="means",
            section_path=("1",),
            alias_target_span=None,
            alias_target_text=None,
        )
        weak_score = _make_candidate_score(
            meaning_id="chunk1-meaning",
            total_score=0.5,
            definition_span=("consulting services", 25, 43),
        )
        weak_resolution = TermResolution(
            occurrence_span=("Services", 80, 88),
            term="Services",
            normalized_key="services",
            chosen_meaning_id="chunk1-meaning",
            chosen_definition_span=("consulting services", 25, 43),
            candidate_scores=(weak_score,),
            resolution_method="tier1",
        )
        result_1 = TermResolutionResult(
            term_meaning_index={"services": (meaning_chunk_1,)},
            meaning_index={"chunk1-meaning": meaning_chunk_1},
            term_resolutions=[weak_resolution],
            ambiguous_keys=("services",),
            undecided=[],
            tier2_report=None,
            tier2_ranked=("rank-a",),
        )

        meaning_chunk_2 = TermMeaning(
            meaning_id="chunk2-meaning",
            surface="Services",
            normalized_key="services",
            ordinal=42,
            intro_span=("Services", 10, 18),
            definition_span=("consulting services", 25, 43),
            definition_text="consulting services",
            intro_kind="means",
            section_path=("1",),
            alias_target_span=None,
            alias_target_text=None,
        )
        strong_score = _make_candidate_score(
            meaning_id="chunk2-meaning",
            total_score=0.9,
            definition_span=("consulting services", 25, 43),
        )
        strong_resolution = TermResolution(
            occurrence_span=("Services", 80, 88),
            term="Services",
            normalized_key="services",
            chosen_meaning_id="chunk2-meaning",
            chosen_definition_span=("consulting services", 25, 43),
            candidate_scores=(strong_score,),
            resolution_method="tier2_blend",
        )
        result_2 = TermResolutionResult(
            term_meaning_index={"services": (meaning_chunk_2,)},
            meaning_index={"chunk2-meaning": meaning_chunk_2},
            term_resolutions=[strong_resolution],
            ambiguous_keys=("services",),
            undecided=[],
            tier2_report={"source": "chunk-2"},
            tier2_ranked=("rank-a", "rank-b"),
        )

        merged = merge_defined_term_results(
            [
                (0, result_1),
                (0, result_2),
            ]
        )

        assert list(merged.meaning_index) == ["term|services|1"]
        merged_meaning = merged.meaning_index["term|services|1"]
        assert merged_meaning.ordinal == 1

        assert merged.term_meaning_index["services"][0].meaning_id == "term|services|1"

        assert len(merged.term_resolutions) == 1
        merged_resolution = merged.term_resolutions[0]
        assert merged_resolution.chosen_meaning_id == "term|services|1"
        assert merged_resolution.resolution_method == "tier2_blend"
        assert merged_resolution.candidate_scores[0].meaning_id == "term|services|1"

        assert merged.ambiguous_keys == ("services",)
        assert merged.tier2_report == {"source": "chunk-2"}
        assert merged.tier2_ranked == ("rank-a", "rank-b")

    def test_shifts_chunk_offsets_before_merging(self) -> None:
        meaning = TermMeaning(
            meaning_id="chunk-meaning",
            surface="Agreement",
            normalized_key="agreement",
            ordinal=1,
            intro_span=("Agreement", 5, 14),
            definition_span=("this agreement", 20, 34),
            definition_text="this agreement",
            intro_kind="means",
            section_path=("2",),
            alias_target_span=None,
            alias_target_text=None,
        )
        resolution = TermResolution(
            occurrence_span=("Agreement", 40, 49),
            term="Agreement",
            normalized_key="agreement",
            chosen_meaning_id="chunk-meaning",
            chosen_definition_span=("this agreement", 20, 34),
            candidate_scores=(),
            resolution_method="tier1",
        )
        result = TermResolutionResult(
            term_meaning_index={"agreement": (meaning,)},
            meaning_index={"chunk-meaning": meaning},
            term_resolutions=[resolution],
            ambiguous_keys=(),
            undecided=[],
            tier2_report=None,
            tier2_ranked=(),
        )

        merged = merge_defined_term_results([(100, result)])

        merged_meaning = merged.meaning_index["term|agreement|1"]
        assert merged_meaning.intro_span == ("Agreement", 105, 114)
        assert merged_meaning.definition_span == ("this agreement", 120, 134)

        merged_resolution = merged.term_resolutions[0]
        assert merged_resolution.occurrence_span == ("Agreement", 140, 149)
        assert merged_resolution.chosen_definition_span == ("this agreement", 120, 134)
