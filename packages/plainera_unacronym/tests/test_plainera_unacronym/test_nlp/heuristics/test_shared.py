import pytest

from plainera_unacronym.nlp.heuristics.shared import has_paren_definition


def _end_of(text: str, token: str) -> int:
    i = text.index(token)
    return i + len(token)

class TestHasParenDefinition:
    def test_detects_basic_definition(self):
        text = "GPU (Graphics Processing Unit) is common."
        end = _end_of(text, "GPU")
        assert has_paren_definition(text, end) is True

    def test_skips_whitespace_before_paren(self):
        text = "CPU   (Central Processing Unit) term."
        end = _end_of(text, "CPU")
        assert has_paren_definition(text, end) is True

    def test_requires_min_letters(self):
        text = "CPU (org) used here."  # only 3 letters
        end = _end_of(text, "CPU")
        assert has_paren_definition(text, end) is False

    def test_no_open_paren_means_false(self):
        text = "CPU - Central Processing Unit"
        end = _end_of(text, "CPU")
        assert has_paren_definition(text, end) is False

    def test_missing_closing_paren_means_false(self):
        text = "CPU (Central Processing Unit"
        end = _end_of(text, "CPU")
        assert has_paren_definition(text, end) is False

    def test_closing_paren_beyond_max_chars_is_false(self):
        inner = "a" * 10  # plenty of letters
        text = f"CPU ({inner}) tail"
        end = _end_of(text, "CPU")
        assert has_paren_definition(text, end, max_chars=5) is False  # ')' too far

    def test_closing_paren_at_boundary_counts_as_true(self):
        inner = "ABCDE"  # 5 letters
        text = f"CPU ({inner}) ok"
        end = _end_of(text, "CPU")
        # '(' at i; we allow j <= i+1+max_chars, so max_chars=5 includes these 5 chars
        assert has_paren_definition(text, end, max_chars=5) is True

    def test_greek_only_is_not_definition(self):
        inner = "αβγδε"
        text = f"ATP ({inner}) binding"
        end = _end_of(text, "ATP")
        assert has_paren_definition(text, end) is False

    def test_ascii_multiword_is_definition(self):
        s = "ATP (adenosine triphosphate) binding"
        assert has_paren_definition(s, s.index("ATP") + 3) is True

    def test_non_letters_inside_do_not_help(self):
        text = "ID (1234-_) stuff"
        end = _end_of(text, "ID")
        assert has_paren_definition(text, end) is False
