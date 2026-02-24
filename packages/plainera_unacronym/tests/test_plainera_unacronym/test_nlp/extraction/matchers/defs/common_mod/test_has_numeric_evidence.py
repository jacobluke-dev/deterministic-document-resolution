import pytest
from plainera_unacronym.nlp.extraction.matchers.defs.common import has_numeric_evidence


class TestHasNumericEvidenceIntegration:
    def test_false_for_letters_only_tokens(self):
        assert has_numeric_evidence(["Portable", "Document", "Format"]) is False

    def test_true_for_numeric_leading_token(self):
        assert has_numeric_evidence(["3M", "Portable", "Format"]) is True
        assert has_numeric_evidence(["2", "Factor", "Auth"]) is True

    def test_true_for_punctuation_wrapped_numeric_leading(self):
        # If your first_alnum_char_upper skips punctuation, these should still count as numeric-leading
        assert has_numeric_evidence(["(3M)", "Portable", "Format"]) is True
        assert has_numeric_evidence(["[2]", "Factor"]) is True
        assert has_numeric_evidence(["'10GbE'", "Link"]) is True

    def test_false_for_tokens_with_no_alnum_anywhere(self):
        assert has_numeric_evidence(["---", "…", "()", ""]) is False

    def test_mixed_tokens_any_numeric_leading_makes_true(self):
        assert has_numeric_evidence(["Alpha", "Beta", "3M", "Gamma"]) is True


class TestHasNumericEvidenceUnit:
    def test_returns_false_when_no_tokens(self):
        assert has_numeric_evidence([]) is False

    def test_returns_false_when_first_alnum_returns_none_for_all(self, _patch):
        # Simulate "no alnum anywhere" for every token
        _patch(has_numeric_evidence, first_alnum_char_upper=lambda tok: None)
        assert has_numeric_evidence(["---", "()", "…"]) is False

    def test_returns_true_when_any_token_has_non_alpha_initial(self, _patch):
        # Map specific tokens to initials
        mapping = {
            "Alpha": "A",  # alpha => not evidence
            "3M": "3",  # digit => evidence
            "Beta": "B",
        }
        _patch(has_numeric_evidence, first_alnum_char_upper=lambda tok: mapping.get(tok))
        assert has_numeric_evidence(["Alpha", "3M", "Beta"]) is True

    def test_returns_false_when_all_initials_are_alpha(self, _patch):
        mapping = {"Alpha": "A", "Beta": "B", "Gamma": "G"}
        _patch(has_numeric_evidence, first_alnum_char_upper=lambda tok: mapping.get(tok))
        assert has_numeric_evidence(["Alpha", "Beta", "Gamma"]) is False

    def test_short_circuits_on_first_numeric_evidence(self, _patch):
        calls = {"n": 0}

        def fake_first_alnum(tok: str):
            calls["n"] += 1
            # 2nd token triggers evidence; 3rd must never be evaluated
            return {"t1": "A", "t2": "2", "t3": "Z"}[tok]

        _patch(has_numeric_evidence, first_alnum_char_upper=fake_first_alnum)

        assert has_numeric_evidence(["t1", "t2", "t3"]) is True
        assert calls["n"] == 2  # proves early-exit

    def test_none_then_numeric_still_true(self, _patch):
        # First token yields None (no alnum), second yields digit => evidence
        mapping = {"junk": None, "2FA": "2"}
        _patch(has_numeric_evidence, first_alnum_char_upper=lambda tok: mapping.get(tok))
        assert has_numeric_evidence(["junk", "2FA"]) is True

    @pytest.mark.parametrize(
        "init, expected",
        [
            ("1", True),
            ("0", True),
            ("9", True),
            ("A", False),
            ("z", False),
            ("_", True),  # non-alpha counts as "numeric evidence" per current logic (not init.isalpha())
            ("-", True),  # same here
        ],
    )
    def test_non_alpha_initial_counts_as_evidence(self, _patch, init, expected):
        _patch(has_numeric_evidence, first_alnum_char_upper=lambda tok: init)
        assert has_numeric_evidence(["anything"]) is expected
