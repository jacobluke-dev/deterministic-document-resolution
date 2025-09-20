import pytest

from plainera_unacronym.nlp.common.constants import APOSTROPHE_VARIANTS
from plainera_unacronym.nlp.common.shared import has_paren_definition, normalize_acronym_key


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


class TestNormalizeKey:
    def test_dotted_mode_strip(self):
        assert normalize_acronym_key("U.S.A.", allow_chars="&-/", dotted_mode="strip") == "USA"

    def test_dotted_mode_preserve(self):
        assert normalize_acronym_key("U.S.A.", allow_chars="&-/", dotted_mode="preserve") == "U.S.A."

    def test_swallow_spaces_ampersand(self):
        assert normalize_acronym_key("R & D", allow_chars="&-/", dotted_mode="strip") == "R&D"
        assert normalize_acronym_key("R& D", allow_chars="&-/", dotted_mode="strip") == "R&D"
        assert normalize_acronym_key("R &D", allow_chars="&-/", dotted_mode="strip") == "R&D"
        assert normalize_acronym_key("R&D", allow_chars="&-/", dotted_mode="strip") == "R&D"

    def test_swallow_spaces_hyphen(self):
        # Single spaces on either/both sides collapse correctly
        assert normalize_acronym_key("GPU - CPU", allow_chars="-&/", dotted_mode="preserve") == "GPU-CPU"
        assert normalize_acronym_key("GPU- CPU", allow_chars="-&/", dotted_mode="preserve") == "GPU-CPU"
        assert normalize_acronym_key("GPU -CPU", allow_chars="-&/", dotted_mode="preserve") == "GPU-CPU"
        assert normalize_acronym_key("GPU-CPU", allow_chars="-&/", dotted_mode="preserve") == "GPU-CPU"

    def test_swallow_spaces_slash(self):
        assert normalize_acronym_key("A / B", allow_chars="/", dotted_mode="strip") == "A/B"
        assert normalize_acronym_key("A/ B", allow_chars="/", dotted_mode="strip") == "A/B"
        assert normalize_acronym_key("A /B", allow_chars="/", dotted_mode="strip") == "A/B"
        assert normalize_acronym_key("A/B", allow_chars="/", dotted_mode="strip") == "A/B"

    def test_non_allowed_separator_keeps_spaces(self):
        # '&' is not allowed here → spaces remain
        assert normalize_acronym_key("R & D", allow_chars="-/", dotted_mode="strip") == "R & D"

    @staticmethod
    def _norm(s: str) -> str:
        return normalize_acronym_key(s, allow_chars="&-/", dotted_mode="preserve")

    @pytest.mark.parametrize("variant", list(APOSTROPHE_VARIANTS.keys()))
    def test_apostrophe_variants_are_canonicalized(self, variant: str) -> None:
        # Every variant becomes ASCII "'"
        assert self._norm(f"O{variant}Reilly") == "O'Reilly"
        assert self._norm(f"rock{variant}n{variant}roll") == "rock'n'roll"
        # Works in all-caps tokens too (your acronym path)
        assert self._norm(f"O{variant}RAN") == "O'RAN"
        # Curly apostrophe should normalize to ASCII "'"
        assert normalize_acronym_key("O’Reilly", allow_chars="&-/", dotted_mode="preserve") == "O'Reilly"


    def test_apostrophe_normalization_is_idempotent(self) -> None:
        assert self._norm("O'Reilly") == "O'Reilly"
        assert self._norm("rock'n'roll") == "rock'n'roll"
        assert self._norm("O'RAN") == "O'RAN"

    def test_dash_variants_are_canonicalized_and_trimmed(self):
        # EN dash / EM dash should map to '-' then spacing rule applies
        assert normalize_acronym_key("GPU – CPU", allow_chars="-", dotted_mode="preserve") == "GPU-CPU"
        assert normalize_acronym_key("A—B", allow_chars="-", dotted_mode="preserve") == "A-B"

    def test_mixed_multiple_allowed_separators(self):
        s = "R & D / E"
        out = normalize_acronym_key(s, allow_chars="&/", dotted_mode="strip")
        assert out == "R&D/E"

    def test_allowed_at_edges(self):
        # Leading/trailing spaces around an allowed separator are swallowed appropriately
        assert normalize_acronym_key("A &B", allow_chars="&", dotted_mode="preserve") == "A&B"
        assert normalize_acronym_key("A& B", allow_chars="&", dotted_mode="preserve") == "A&B"
