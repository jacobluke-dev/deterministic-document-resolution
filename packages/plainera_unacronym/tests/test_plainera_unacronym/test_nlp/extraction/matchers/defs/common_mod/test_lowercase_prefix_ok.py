import pytest

from plainera_unacronym.nlp.extraction.matchers.defs.common import _lowercase_prefix_ok


class TestLowercasePrefixOkUnit:
    def test_disabled_flag_returns_false(self, _patch):
        _patch(_lowercase_prefix_ok, PUNCT_TRIM="", is_mixed_case_acronym=lambda acr: True)
        assert _lowercase_prefix_ok(
            acr="iOS", tokens=["iPhone"], token_idx=0, acr_pos=0, lowercase_prefix_exception=False
        ) is False

    def test_only_applies_at_acr_pos0_and_token0(self, _patch):
        _patch(_lowercase_prefix_ok, PUNCT_TRIM="", is_mixed_case_acronym=lambda acr: True)

        assert _lowercase_prefix_ok(
            acr="iOS", tokens=["iPhone"], token_idx=1, acr_pos=0, lowercase_prefix_exception=True
        ) is False
        assert _lowercase_prefix_ok(
            acr="iOS", tokens=["iPhone"], token_idx=0, acr_pos=2, lowercase_prefix_exception=True
        ) is False

    def test_requires_mixed_case_acronym(self, _patch):
        _patch(_lowercase_prefix_ok, PUNCT_TRIM="", is_mixed_case_acronym=lambda acr: False)
        assert _lowercase_prefix_ok(
            acr="IOS", tokens=["iPhone"], token_idx=0, acr_pos=0, lowercase_prefix_exception=True
        ) is False

    def test_rejects_allcaps_alpha_token0(self, _patch):
        _patch(_lowercase_prefix_ok, PUNCT_TRIM="", is_mixed_case_acronym=lambda acr: True)
        assert _lowercase_prefix_ok(
            acr="iOS", tokens=["HTTP"], token_idx=0, acr_pos=0, lowercase_prefix_exception=True
        ) is False

    def test_accepts_prefix_match_on_token0(self, _patch):
        _patch(_lowercase_prefix_ok, PUNCT_TRIM="", is_mixed_case_acronym=lambda acr: True)
        assert _lowercase_prefix_ok(
            acr="iOS", tokens=["iPhone"], token_idx=0, acr_pos=0, lowercase_prefix_exception=True
        ) is True

    def test_prefix_match_uses_punct_trim(self, _patch):
        _patch(_lowercase_prefix_ok, PUNCT_TRIM="()\"", is_mixed_case_acronym=lambda acr: True)
        assert _lowercase_prefix_ok(
            acr="mRNA", tokens=['("molecule")'], token_idx=0, acr_pos=0, lowercase_prefix_exception=True
        ) is True


class TestLowerCasePrefixOkIntegration:
    def test_prefix_match_false_when_exception_disabled(self):
        assert _lowercase_prefix_ok(
            acr="mRNA", tokens=['("molecule")'], token_idx=0, acr_pos=0, lowercase_prefix_exception=False
        ) is False

    def test_prefix_match_false_when_not_first_acr_char(self):
        assert _lowercase_prefix_ok(
            acr="mRNA", tokens=['("molecule")'], token_idx=0, acr_pos=1, lowercase_prefix_exception=True
        ) is False

    def test_prefix_match_false_when_not_first_token(self):
        assert _lowercase_prefix_ok(
            acr="mRNA", tokens=['x', '("molecule")'], token_idx=1, acr_pos=0, lowercase_prefix_exception=True
        ) is False

    def test_prefix_match_false_when_token0_is_allcaps_word(self):
        # blocks mapping lowercase prefix onto an all-caps word token
        assert _lowercase_prefix_ok(
            acr="mRNA", tokens=["MOLECULE"], token_idx=0, acr_pos=0, lowercase_prefix_exception=True
        ) is False

    def test_prefix_match_false_when_first_char_does_not_match(self):
        assert _lowercase_prefix_ok(
            acr="mRNA", tokens=["protein"], token_idx=0, acr_pos=0, lowercase_prefix_exception=True
        ) is False


class TestLowercasePrefixOkEdges:
    def test_allows_lowercase_prefix_on_token0_even_if_not_alpha(self, _patch):
        _patch(_lowercase_prefix_ok, is_mixed_case_acronym=lambda acr: True, PUNCT_TRIM="")

        assert _lowercase_prefix_ok(
            acr="iOS",
            tokens=["iOS-compatible"],
            token_idx=0,
            acr_pos=0,
            lowercase_prefix_exception=True,
        ) is True

    def test_empty_token_after_trim_returns_false(self, _patch):
        _patch(_lowercase_prefix_ok, is_mixed_case_acronym=lambda acr: True, PUNCT_TRIM=".")

        assert _lowercase_prefix_ok(
            acr="iOS",
            tokens=["..."],
            token_idx=0,
            acr_pos=0,
            lowercase_prefix_exception=True,
        ) is False
