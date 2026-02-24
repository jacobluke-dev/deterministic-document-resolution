import pytest
from plainera_unacronym.nlp.extraction.matchers.defs.common import is_acronym_like_token


class TestIsAcronymLikeToken:
    @pytest.mark.parametrize(
        "tok",
        [
            "RNA",
            "HTTP2",
            "U.S.A",
            "U.S.A.",
            "HTTP,",
            'HTTP"',
            "HTTP)",
            "HTTP»",
            "X1",
        ],
    )
    def test_true_for_common_acronym_like_forms_and_trailing_punct(self, tok):
        assert is_acronym_like_token(tok) is True

    @pytest.mark.parametrize(
        "tok",
        [
            "",  # empty
            "A",  # too short after trim
            "a",  # too short / lowercase
            "Pdf",  # has lowercase
            "foo",  # all lowercase
            "1234",  # no alpha chars
            "....",  # trims to empty-ish / no alpha
            "A.",  # length becomes < 2 after trim
            "U.S.a",  # contains lowercase
        ],
    )
    def test_false_for_non_acronym_like_tokens(self, tok):
        assert is_acronym_like_token(tok) is False

    def test_true_for_uppercase_with_digits_and_no_lowercase(self):
        assert is_acronym_like_token("X509") is True

    def test_false_when_contains_any_lowercase_even_if_has_uppercase(self):
        assert is_acronym_like_token("HTTPs") is False
        assert is_acronym_like_token("mRNA") is False

    def test_trims_only_trailing_punct_not_internal(self):
        # internal punctuation isn't stripped by .strip(...), but dotted initialisms are still supported
        assert is_acronym_like_token("U.S.A") is True
        assert is_acronym_like_token("U..S") is True  # doesn't match your regex

    def test_contains_lowercase_is_not_acronym_like(self):
        # violates fallback: contains lowercase
        assert is_acronym_like_token("U..s") is False
