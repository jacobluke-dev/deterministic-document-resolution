from plainera_unacronym.nlp.detection.defined_terms.builders import (
    build_defined_term_occurrence,
    build_defined_term_sense,
)


class TestBuildDefinedTermSense:
    def test_build_defined_term_sense_sets_expected_fields(self):
        result = build_defined_term_sense(
            term='"Effective Date"',
            term_start=10,
            term_end=26,
            provenance="defined_term_detector",
        )

        assert result.term == "Effective Date"
        assert result.start_offset == 10
        assert result.end_offset == 26
        assert result.normalized_key == "effective_date"
        assert result.provenance == "defined_term_detector"

    def test_build_defined_term_sense_strips_trailing_punctuation_and_quotes(self):
        result = build_defined_term_sense(
            term='"Confidential Information."',
            term_start=50,
            term_end=77,
            provenance="defined_term_detector",
        )

        assert result.term == "Confidential Information"
        assert result.normalized_key == "confidential_information"

    def test_build_defined_term_sense_normalizes_bridge_words(self):
        result = build_defined_term_sense(
            term="Change of Control",
            term_start=100,
            term_end=117,
            provenance="defined_term_detector",
        )

        assert result.term == "Change of Control"
        assert result.normalized_key == "change_of_control"


class TestBuildDefinedTermOccurrence:
    def test_build_defined_term_occurrence_sets_expected_fields(self):
        result = build_defined_term_occurrence(
            term='"Services"',
            start_offset=200,
            end_offset=210,
            segment_window="...the Services from the Effective Date...",
            confidence=0.85,
        )

        assert result.term == "Services"
        assert result.start_offset == 200
        assert result.end_offset == 210
        assert result.normalized_key == "services"
        assert result.occurrence_confidence == 0.85
        assert result.segment_window == "...the Services from the Effective Date..."

    def test_build_defined_term_occurrence_strips_trailing_punctuation_and_quotes(self):
        result = build_defined_term_occurrence(
            term='"Effective Date."',
            start_offset=300,
            end_offset=317,
        )

        assert result.term == "Effective Date"
        assert result.normalized_key == "effective_date"

    def test_build_defined_term_occurrence_defaults_confidence_and_segment_window(self):
        result = build_defined_term_occurrence(
            term="Change of Control",
            start_offset=400,
            end_offset=417,
        )

        assert result.term == "Change of Control"
        assert result.normalized_key == "change_of_control"
        assert result.occurrence_confidence == 1.0
        assert result.segment_window is None
