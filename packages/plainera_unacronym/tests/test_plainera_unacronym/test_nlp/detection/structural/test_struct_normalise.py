from __future__ import annotations

from plainera_unacronym.nlp.detection.structural import normalize_structural_reference_key


class TestNormalizeStructuralReferenceKey:
    def test_normalize_structural_reference_key_schedule_alpha(self):
        out = normalize_structural_reference_key("Schedule", "A")

        assert out == "schedule_a"

    def test_normalize_structural_reference_key_section_decimal(self):
        out = normalize_structural_reference_key("Section", "4.2")

        assert out == "section_4_2"

    def test_normalize_structural_reference_key_article_roman(self):
        out = normalize_structural_reference_key("Article", "III")

        assert out == "article_iii"

    def test_normalize_structural_reference_key_trims_and_lowercases(self):
        out = normalize_structural_reference_key("  Appendix  ", "  C  ")

        assert out == "appendix_c"

    def test_normalize_structural_reference_key_collapses_internal_whitespace(self):
        out = normalize_structural_reference_key("Schedule", "  10   A  ")

        assert out == "schedule_10_a"

    def test_normalize_structural_reference_key_removes_punctuation_noise(self):
        out = normalize_structural_reference_key("Clause:", "7.3,")

        assert out == "clause_7_3"

    def test_normalize_structural_reference_key_handles_hyphenated_kind(self):
        out = normalize_structural_reference_key("Sub-Section", "4.2")

        assert out == "sub-section_4_2"
