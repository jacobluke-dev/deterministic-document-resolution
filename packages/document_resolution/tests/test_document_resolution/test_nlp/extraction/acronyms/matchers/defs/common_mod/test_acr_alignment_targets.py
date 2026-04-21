from document_resolution.nlp.extraction.acronyms.matchers.defs.defs_common import acr_alignment_targets


class TestAcrAlignmentTargets:
    def test_returns_empty_when_no_alnum(self):
        assert acr_alignment_targets("()---", has_numeric_evidence=False) == []
        assert acr_alignment_targets("…—", has_numeric_evidence=True) == []

    def test_keeps_digits_when_numeric_evidence_true(self):
        # Digits preserved
        assert acr_alignment_targets("E2E", has_numeric_evidence=True) == ["E", "2", "E"]
        assert acr_alignment_targets("10GbE", has_numeric_evidence=True) == ["1", "0", "G", "b", "E"]

    def test_drops_digits_when_numeric_evidence_false(self):
        # Digits dropped (optional) => letters only
        assert acr_alignment_targets("E2E", has_numeric_evidence=False) == ["E", "E"]
        assert acr_alignment_targets("10GbE", has_numeric_evidence=False) == ["G", "b", "E"]

    def test_strips_non_alnum_but_preserves_letter_case(self):
        # Only alnum kept; case preserved from input
        assert acr_alignment_targets("mRNA-1", has_numeric_evidence=True) == ["m", "R", "N", "A", "1"]
        assert acr_alignment_targets("mRNA-1", has_numeric_evidence=False) == ["m", "R", "N", "A"]
