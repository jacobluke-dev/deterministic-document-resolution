from document_resolution.nlp.detection.defined_terms.builders import (
    build_defined_term_intro,
    build_defined_term_mention,
)


class TestBuildDefinedTermIntro:
    def test_build_defined_term_intro_sets_expected_fields(self):
        result = build_defined_term_intro(
            term='"Effective Date"',
            term_start=10,
            term_end=26,
            provenance="defined_term_detector",
            intro_kind="quoted_means"
        )

        assert result.term == "Effective Date"
        assert result.start_offset == 10
        assert result.end_offset == 26
        assert result.normalized_key == "effective_date"
        assert result.provenance == "defined_term_detector"

    def test_build_defined_term_intro_strips_trailing_punctuation_and_quotes(self):
        result = build_defined_term_intro(
            term='"Confidential Information."',
            term_start=50,
            term_end=77,
            provenance="defined_term_detector",
            intro_kind="quoted_means"
        )

        assert result.term == "Confidential Information"
        assert result.normalized_key == "confidential_information"

    def test_build_defined_term_intro_normalizes_bridge_words(self):
        result = build_defined_term_intro(
            term="Change of Control",
            term_start=100,
            term_end=117,
            provenance="defined_term_detector",
            intro_kind="quoted_means"
        )

        assert result.term == "Change of Control"
        assert result.normalized_key == "change_of_control"


class TestBuildDefinedTermMention:
    def test_build_defined_term_mention_sets_expected_fields(self):
        result = build_defined_term_mention(
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
        assert result.confidence == 0.85
        assert result.segment_window == "...the Services from the Effective Date..."

    def test_build_defined_term_mention_strips_trailing_punctuation_and_quotes(self):
        result = build_defined_term_mention(
            term='"Effective Date."',
            start_offset=300,
            end_offset=317,
        )

        assert result.term == "Effective Date"
        assert result.normalized_key == "effective_date"

    def test_build_defined_term_mention_defaults_confidence_and_segment_window(self):
        result = build_defined_term_mention(
            term="Change of Control",
            start_offset=400,
            end_offset=417,
        )

        assert result.term == "Change of Control"
        assert result.normalized_key == "change_of_control"
        assert result.confidence == 1.0
        assert result.segment_window is None
