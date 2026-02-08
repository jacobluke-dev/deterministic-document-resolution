import unicodedata

import pytest
from plainera_unacronym.nlp.common.constants_regex import APOSTROPHE_VARIANTS
from plainera_unacronym.nlp.common.shared import (
    _swallow_spaces_around_allowed,
    canonicalize,
    collapse_ws,
    has_letter,
    has_paren_definition,
    normalize_acronym_key,
    strip_trailing_punct_str,
)
from plainera_unacronym.nlp.extraction.anchored.normalise import tighten_definition_span


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

class TestCanonicalize:
    def test_apostrophe_variants_fold_to_ascii_quote(self):
        assert canonicalize("don’t") == "don't"      # U+2019
        assert canonicalize("don‘ t") == "don' t"    # U+2018
        assert canonicalize("rockʼnʼroll") == "rock'n'roll"  # U+02BC

    def test_dash_variants_fold_to_ascii_hyphen(self):
        assert canonicalize("A–B") == "A-B"          # en dash
        assert canonicalize("A—B") == "A-B"          # em dash

    def test_nfkc_collapses_fullwidth_apostrophe_then_translates(self):
        # Fullwidth apostrophe U+FF07 maps to ASCII via NFKC + translation.
        assert canonicalize("＇quote＇") == "'quote'"

    def test_preserves_plain_ascii(self):
        s = "simple - 'ascii' text"
        assert canonicalize(s) == s

    def test_nfkc_is_applied(self):
        # This asserts behaviour without relying on your specific mapping table.
        # Ligature 'ﬁ' (U+FB01) typically NFKC-normalises to "fi".
        assert canonicalize("ﬁ") == unicodedata.normalize("NFKC", "ﬁ")


class TestStripTrailingPunctStr:
    def test_strips_single_trailing_punctuation(self):
        assert strip_trailing_punct_str("RNA,") == "RNA"
        assert strip_trailing_punct_str("word.") == "word"
        assert strip_trailing_punct_str("ok!") == "ok"

    def test_strips_multiple_trailing_punctuation(self):
        assert strip_trailing_punct_str("hello!!!") == "hello"
        assert strip_trailing_punct_str("what?!") == "what"
        assert strip_trailing_punct_str("end»”") == "end"

    def test_strips_trailing_whitespace(self):
        assert strip_trailing_punct_str("RNA   ") == "RNA"
        assert strip_trailing_punct_str("RNA,\n\t ") == "RNA"

    def test_does_not_strip_leading_or_internal_punctuation(self):
        assert strip_trailing_punct_str("(RNA") == "(RNA"
        assert strip_trailing_punct_str("co-op") == "co-op"
        assert strip_trailing_punct_str("a,b") == "a,b"

    def test_strips_closing_brackets_and_braces(self):
        assert strip_trailing_punct_str("Unit)") == "Unit"
        assert strip_trailing_punct_str("Thing]}") == "Thing"

    def test_strip_trailing_punct_variants_agree_on_terminal_dot(self):
        s = "U.S.A.)"
        assert strip_trailing_punct_str(s) == "U.S.A"


class TestSwallowSpacesAroundAllowed:
    def test_returns_input_unchanged_when_allow_chars_empty(self):
        assert _swallow_spaces_around_allowed("R & D", "") == "R & D"
        assert _swallow_spaces_around_allowed("  R  ", "") == "  R  "

    def test_collapses_spaces_around_ampersand(self):
        assert _swallow_spaces_around_allowed("R & D", "&") == "R&D"
        assert _swallow_spaces_around_allowed("R  &   D", "&") == "R&D"

    def test_swallow_spaces_on_left_of_allowed_char(self):
        assert _swallow_spaces_around_allowed("R &D", "&") == "R&D"
        assert _swallow_spaces_around_allowed("R  &D", "&") == "R&D"

    def test_swallow_spaces_on_right_of_allowed_char(self):
        assert _swallow_spaces_around_allowed("R& D", "&") == "R&D"
        assert _swallow_spaces_around_allowed("R&   D", "&") == "R&D"

    def test_multiple_allowed_chars_are_supported(self):
        assert _swallow_spaces_around_allowed("A / B", "/&") == "A/B"
        assert _swallow_spaces_around_allowed("A & B", "/&") == "A&B"

    def test_does_not_modify_spaces_unrelated_to_allowed_chars(self):
        assert _swallow_spaces_around_allowed("hello world", "&") == "hello world"
        assert _swallow_spaces_around_allowed("A - B", "&") == "A - B"

    def test_allow_chars_are_treated_literally(self):
        # Ensure regex escaping works for special characters like '+'.
        assert _swallow_spaces_around_allowed("mRNA + seq", "+") == "mRNA+seq"


class TestNormalizeAcronymKeyIntegration:
    def test_nfkc_and_quote_dash_folding(self):
        # fullwidth apostrophe -> ASCII apostrophe (via NFKC + translate)
        assert normalize_acronym_key("＇ABC＇", allow_chars="", dotted_mode="preserve") == "'ABC'"
        # em dash -> hyphen
        assert normalize_acronym_key("A—B", allow_chars="-", dotted_mode="preserve") == "A-B"

    def test_dotted_strip_removes_periods(self):
        assert normalize_acronym_key("U.S.A.", allow_chars=".", dotted_mode="strip") == "USA"
        assert normalize_acronym_key("R.N.A", allow_chars=".", dotted_mode="strip") == "RNA"

    def test_dotted_preserve_keeps_periods(self):
        assert normalize_acronym_key("U.S.A.", allow_chars=".", dotted_mode="preserve") == "U.S.A."
        assert normalize_acronym_key("R.N.A", allow_chars=".", dotted_mode="preserve") == "R.N.A"

    def test_swallow_spaces_around_allowed_connectors_only(self):
        assert normalize_acronym_key("R & D", allow_chars="&", dotted_mode="preserve") == "R&D"
        assert normalize_acronym_key("A / B", allow_chars="/", dotted_mode="preserve") == "A/B"

        # Not allowed -> must remain unchanged (spaces preserved)
        assert normalize_acronym_key("R & D", allow_chars="", dotted_mode="preserve") == "R & D"

    def test_preserves_case(self):
        assert normalize_acronym_key("mRNA", allow_chars="", dotted_mode="preserve") == "mRNA"
        assert normalize_acronym_key("RNA", allow_chars="", dotted_mode="preserve") == "RNA"

    def test_combined_behaviour_dots_and_connectors(self):
        # Dots stripped first, then connector whitespace collapse.
        assert normalize_acronym_key("R. &  D.", allow_chars="&.", dotted_mode="strip") == "R&D"

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
        # Works in all-caps tokens too (the acronym path)
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



    def test_key_dotted_and_separators(self):
        assert normalize_acronym_key("U.S.A.", "&-./", "strip") == "USA"
        assert normalize_acronym_key("R & D", "&-./", "strip") == "R&D"
        assert normalize_acronym_key("R – D", "&-./", "strip") == "R-D"  # dash folded


class TestCollapseWs:
    def test_collapses_internal_whitespace(self):
        assert collapse_ws("a   b\t\tc") == "a b c"

    def test_trims_leading_and_trailing_whitespace(self):
        assert collapse_ws("   hello world  ") == "hello world"

    def test_handles_newlines(self):
        assert collapse_ws("a\nb\r\nc") == "a b c"

class TestTightenDefinitionSpan:
    def test_keeps_titlecase_with_per(self):
        s = "Cost per Acquisition"
        out = tighten_definition_span(s)
        assert out == "Cost per Acquisition"

    def test_keeps_titlecase_with_common_linkers(self):
        s = "Department of Health and Social Care"
        out = tighten_definition_span(s)
        assert out == "Department of Health and Social Care"

    def test_keeps_ampersand(self):
        s = "Research & Development"
        out = tighten_definition_span(s)
        assert out == "Research & Development"

    def test_trailing_comma_is_ignored(self):
        s = "Cost per Acquisition,"
        out = tighten_definition_span(s)
        assert out == "Cost per Acquisition"

    def test_all_caps_phrase_is_preserved(self):
        s = "COST PER ACQUISITION"
        out = tighten_definition_span(s)
        assert out == "COST PER ACQUISITION"

    def test_prefers_last_titlecase_run_at_end(self):
        s = "Some intro text, Department of Education and Skills"
        out = tighten_definition_span(s)
        assert out == "Department of Education and Skills"

    def test_fallback_when_no_titlecase_run(self):
        s = "this is a lowercase tail with numbers 123"
        out = tighten_definition_span(s)
        assert out == "this is a lowercase tail with numbers 123"

    def test_handles_unicode_apostrophes_and_dashes(self):
        s = "Director-General’s Office – North"
        # We end the string with a TitleCase run so it’s selected
        s2 = f"See memo for {s}"
        out = tighten_definition_span(s2)
        assert out == "Director-General’s Office – North"

    def test_works_when_titlecase_is_after_a_boundary(self):
        # Even if the BOUNDARY_RE logic changes, ending with the TitleCase run keeps this robust
        s = "some preface. Cost per Acquisition"
        out = tighten_definition_span(s)
        assert out == "Cost per Acquisition"

    def test_picks_titlecase_run_for_pto_sentence(self):
        s = "Please Turn Over on print jobs."
        out = tighten_definition_span(s)
        assert out == "Please Turn Over"


class TestHasLetters:

    @pytest.mark.parametrize(
        "s,expected",
        [
            ("-_/.,;:!?()", False),
            ("12345", False),
            ("SSO", True),
            ("single sign-on", True),
            ("O'Neil", True),
            ("", False),  # empty
            ("   \t", False),  # whitespace only
            ("123456", False),  # digits
            ("--._", False),  # punctuation/symbols
            ("\u0301", False),  # combining acute accent (not a letter)
            ("🧠💡", False),  # emoji
            ("A", True),  # ASCII letter
            ("abc123", True),  # mixed alnum
            ("42 is the answer", True),  # sentence with letters
            ("Straße", True),  # Latin letter ß
            ("Ångström", True),  # Latin with diacritics
            ("中文", True),  # CJK
            ("Ж9", True),  # Cyrillic + digit
            ("β-blocker", True),  # Greek + hyphen
        ],
    )
    def test_various_inputs(self, s, expected):
        assert has_letter(s) is expected

    def test_long_string_performance_smoke(self):
        s = "1234567" * 1000 + "X" + "!" * 1000
        assert has_letter(s) is True


    @pytest.mark.parametrize(
        "s,expected",
        [
            ("abc", True),
            ("ABC", True),
            ("a1!", True),  # mixed, has a letter
            ("", False),
            ("123", False),  # digits only
            ("!!!", False),  # punctuation only
            (" \t\n", False),  # whitespace only
            ("   A   ", True),  # letters among spaces
            ("é", True),  # accented letter
            ("ß", True),  # Unicode letter
            ("Δ", True),  # Greek letter
            ("中", True),  # CJK letter
            ("🙂", False),  # emoji is not alpha
        ],
    )
    def test_various_strings(self, s, expected):
        assert has_letter(s) is expected
