from __future__ import annotations

from plainera_unacronym.nlp.detection.structural.builders import build_structural_reference


class TestBuildStructuralReference:
    def test_build_structural_reference_returns_expected_model(self):
        out = build_structural_reference(
            kind="Section",
            label="4.2",
            start_offset=10,
            end_offset=21,
            provenance="structural_reference_detector",
        )

        assert out.kind == "Section"
        assert out.label == "4.2"
        assert out.start_offset == 10
        assert out.end_offset == 21
        assert out.normalized_key == "section_4_2"
        assert out.provenance == "structural_reference_detector"

    def test_build_structural_reference_strips_trailing_punctuation(self):
        out = build_structural_reference(
            kind="Clause:",
            label="7.3,",
            start_offset=0,
            end_offset=11,
            provenance="structural_reference_detector",
        )

        assert out.kind == "Clause"
        assert out.label == "7.3"
        assert out.normalized_key == "clause_7_3"

    def test_build_structural_reference_trims_surrounding_whitespace(self):
        out = build_structural_reference(
            kind="  Schedule  ",
            label="  A  ",
            start_offset=5,
            end_offset=15,
            provenance="structural_reference_detector",
        )

        assert out.kind == "Schedule"
        assert out.label == "A"
        assert out.normalized_key == "schedule_a"

    def test_build_structural_reference_normalises_roman_label(self):
        out = build_structural_reference(
            kind="Article",
            label="III",
            start_offset=20,
            end_offset=31,
            provenance="structural_reference_detector",
        )

        assert out.kind == "Article"
        assert out.label == "III"
        assert out.normalized_key == "article_iii"

    def test_build_structural_reference_preserves_provenance(self):
        out = build_structural_reference(
            kind="Appendix",
            label="C",
            start_offset=3,
            end_offset=13,
            provenance="test_source",
        )

        assert out.provenance == "test_source"
