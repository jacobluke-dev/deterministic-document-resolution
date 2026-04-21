import document_resolution.nlp.detection.heuristics.core as core
import pytest
from document_resolution.nlp.common.constants_regex import TRAILING_PUNCT_CHARS
from document_resolution.nlp.detection.heuristics.core import (
    _contained_in_any,
    caps_ratio,
    core_len_for_bounds,
    has_stands_for_follow,
    in_brackets,
    letters,
    next_word_lowercase,
    prev_token,
    strip_trailing_punct_span,
    threshold_len,
)


def _end(text: str, token: str) -> int:
    s = text.index(token)
    return s + len(token)


def _start_of(text: str, token: str) -> int:
    return text.index(token)


class TestLetters:
    def test_ascii_letters_only(self):
        assert letters("abcXYZ") == "abcXYZ"

    def test_strips_digits_and_punct(self):
        assert letters("H2O") == "HO"
        assert letters("U.S.A.") == "USA"
        assert letters("rock'n'roll") == "rocknroll"

    def test_strips_whitespace(self):
        assert letters("  a b  c ") == "abc"
        assert letters("\tAlpha\nBeta\r") == "AlphaBeta"

    def test_empty_and_nonletters_only(self):
        assert letters("") == ""
        assert letters("1234-_.:,()") == ""
        assert letters("🙂🚀") == ""

    def test_unicode_accents_and_scripts(self):
        assert letters("Éclair") == "Éclair"  # accented Latin preserved
        assert letters("İi") == "İi"  # Turkish dotted I + i
        assert letters("Δx") == "Δx"  # Greek + Latin
        assert letters("中A3!") == "中A"  # CJK + Latin; digits/punct dropped


class TestCapsRatio:
    def test_all_upper_is_1(self):
        assert caps_ratio("GPU") == 1.0

    def test_all_lower_is_0(self):
        assert caps_ratio("gpu") == 0.0

    def test_mixed_ascii(self):
        # N a C l -> 2/4 uppercase
        assert caps_ratio("NaCl") == 0.5

    def test_digits_and_punct_ignored(self):
        # Letters: H,O -> both uppercase => 1.0
        assert caps_ratio("H2O") == 1.0
        # Letters: U,S,A -> all uppercase => 1.0
        assert caps_ratio("U.S.A.") == 1.0
        # Letters: rocknroll (apostrophes ignored) -> all lower => 0.0
        assert caps_ratio("rock'n'roll") == 0.0

    def test_no_letters_returns_0(self):
        assert caps_ratio("") == 0
        assert caps_ratio("1234-._") == 0

    def test_unicode_accented(self):
        # Letters: É c l a i r -> 1/6 uppercase
        assert caps_ratio("Éclair") == pytest.approx(1 / 6)

    def test_unicode_turkish_and_greek(self):
        # İ i -> 1/2 uppercase
        assert caps_ratio("İi") == 0.5
        # Δ x -> 1/2 uppercase
        assert caps_ratio("Δx") == 0.5

    def test_unicode_cjk_with_latin(self):
        # Letters: 中 (caseless, not upper) + A (upper) -> 1/2
        assert caps_ratio("中A3!") == 0.5


class TestStripTrailingPunct:
    def test_no_trailing_punct_no_change(self, span):
        text = "Alpha GPU Beta"
        s, e = span(text, "GPU")
        ns, ne = strip_trailing_punct_span(text, s, e)
        assert (ns, ne) == (s, e)
        assert text[ns:ne] == "GPU"

    def test_strip_single_period(self):
        text = "Memory uses RAM."
        # include '.' in the span
        s = text.index("RAM")
        e = s + len("RAM.")
        ns, ne = strip_trailing_punct_span(text, s, e)
        assert text[ns:ne] == "RAM"
        # the '.' should now be just outside the slice
        assert text[ne : ne + 1] == "."

    def test_strip_multiple_closers_chain(self):
        text = "Token!?) next"
        # span includes all three trailing punct chars
        s = text.index("Token")
        e = s + len("Token!?)")
        ns, ne = strip_trailing_punct_span(text, s, e)
        # all trailing punct removed, leaving bare token
        assert text[ns:ne] == "Token"

    def test_only_punctspan_becomes_empty(self):
        text = "Hello !!! there"
        s = text.index("!!!")
        e = s + 3
        ns, ne = strip_trailing_punct_span(text, s, e)
        assert ns == ne  # empty slice after stripping

    def test_parametric_known_trailing_chars(self):
        # Verify behavior for whatever is actually configured in TRAILING_PUNCT_DEFAULT
        base = "ACRONYM"
        for ch in [".", "!", "?", ")", "]", "'", '"', "”", ",", ";", ":"]:
            text = base + ch + " tail"
            s = 0
            e = len(base) + 1  # include the trailing char
            ns, ne = strip_trailing_punct_span(text, s, e)
            if ch in TRAILING_PUNCT_CHARS:
                assert text[ns:ne] == base
            else:
                assert text[ns:ne] == base + ch


class TestInBrackets:
    def test_no_brackets(self, span):
        text = "foo GPU bar"
        s, e = span(text, "GPU")
        assert in_brackets(text, s, e) == (False, False)

    def test_inside_parentheses(self, span):
        text = "(GPU)"
        s, e = span(text, "GPU")
        # inside True implies adjacent True as well
        assert in_brackets(text, s, e) == (True, True)

    def test_inside_square_brackets(self, span):
        text = "[GPU]"
        s, e = span(text, "GPU")
        assert in_brackets(text, s, e) == (True, True)

    def test_adjacent_curly_quotes_only(self, span):
        text = "“GPU”"
        s, e = span(text, "GPU")
        # Curly quotes are considered adjacent, not "inside"
        assert in_brackets(text, s, e) == (False, True)

    def test_adjacent_left_only(self, span):
        text = "«GPU token"
        s, e = span(text, "GPU")
        assert in_brackets(text, s, e) == (False, True)

    def test_adjacent_right_only(self, span):
        text = "GPU»"
        s, e = span(text, "GPU")
        assert in_brackets(text, s, e) == (False, True)

    def test_start_of_text_right_bracket(self, span):
        text = "GPU)"
        s, e = span(text, "GPU")
        # s == 0 → cannot be "inside", but right bracket makes it adjacent
        assert in_brackets(text, s, e) == (False, True)

    def test_end_of_text_left_bracket(self, span):
        text = "(GPU"
        s, e = span(text, "GPU")
        # e == len(text) → cannot be "inside", but left bracket makes it adjacent
        assert in_brackets(text, s, e) == (False, True)

    def test_nested_brackets_inside(self, span):
        text = "((GPU))"
        s, e = span(text, "GPU")
        assert in_brackets(text, s, e) == (True, True)

    def test_mismatched_pair_counts_as_inside_by_current_rule(self, span):
        text = "[GPU)"
        s, e = span(text, "GPU")
        # Left '[' and right ')' both satisfy inside-condition in current implementation
        assert in_brackets(text, s, e) == (True, True)


class TestHasStandsForFollow:
    def test_positive_basic(self):
        text = "We use GPU stands for Graphics Processing Unit."
        assert has_stands_for_follow(text, _end(text, "GPU")) is True

    def test_case_insensitive(self):
        text = "Abbrev GPU STANDS FOR Graphics."
        assert has_stands_for_follow(text, _end(text, "GPU")) is True

    def test_respects_sentence_boundary_period(self):
        text = "Abbrev GPU. stands for Graphics."
        # Stops at '.' immediately after GPU → no scan range → False
        assert has_stands_for_follow(text, _end(text, "GPU")) is False

    def test_respects_sentence_boundary_exclaim_question(self):
        assert has_stands_for_follow("GPU! stands for X", _end("GPU! stands for X", "GPU")) is False
        assert has_stands_for_follow("GPU? stands for X", _end("GPU? stands for X", "GPU")) is False

    def test_max_chars_cutoff_false_when_too_far(self):
        text = "GPU " + ("x" * 20) + " stands for Graphics."
        # Place 'stands for' beyond a tight limit
        assert has_stands_for_follow(text, _end(text, "GPU"), max_chars=10) is False

    def test_max_chars_within_limit_true(self):
        text = "GPU xxx stands for Graphics."
        assert has_stands_for_follow(text, _end(text, "GPU"), max_chars=24) is True

    def test_comma_does_not_stop_scan(self):
        text = "GPU, stands for Graphics."
        # Comma is not a terminator for this function
        assert has_stands_for_follow(text, _end(text, "GPU")) is True

    def test_quotes_do_not_stop_scan(self):
        text = 'GPU "stands for" greatness.'
        assert has_stands_for_follow(text, _end(text, "GPU")) is True

    def test_no_match(self):
        text = "GPU standard format pipeline."
        assert has_stands_for_follow(text, _end(text, "GPU")) is False


class TestNextWordLowercase:
    def test_basic_lowercase(self):
        text = "GPU stands for speed."
        assert next_word_lowercase(text, _end(text, "GPU")) is True

    def test_capitalized_is_false(self):
        text = "GPU Graphs are nice."
        assert next_word_lowercase(text, _end(text, "GPU")) is False

    def test_mixed_case_is_false(self):
        text = "GPU iOS builds are weekly."
        assert next_word_lowercase(text, _end(text, "GPU")) is False

    def test_skips_spaces_and_simple_quote(self):
        text = "GPU   'graphics' pipeline"
        assert next_word_lowercase(text, _end(text, "GPU")) is True  # -> graphics

    def test_skips_curly_quotes_and_brackets(self):
        text = "GPU “graphics” stage"
        assert next_word_lowercase(text, _end(text, "GPU")) is True
        text2 = "Token ([alpha]) beta"
        assert next_word_lowercase(text2, _end(text2, "Token")) is True  # -> alpha

    def test_handles_apostrophes_inside_word(self):
        text = "GPU rock'n'roll forever"
        assert next_word_lowercase(text, _end(text, "GPU")) is True

    def test_next_is_punctuation_yields_false(self):
        text = "GPU, then proceed"
        assert next_word_lowercase(text, _end(text, "GPU")) is False

    def test_next_is_number_yields_false(self):
        text = "GPU 123abc later"
        assert next_word_lowercase(text, _end(text, "GPU")) is False

    def test_unicode_lowercase(self):
        text = "GPU café tests"
        assert next_word_lowercase(text, _end(text, "GPU")) is True

    def test_end_of_text(self):
        text = "GPU"
        assert next_word_lowercase(text, _end(text, "GPU")) is False


class TestPrevToken:
    def test_basic_previous_word(self):
        text = "foo bar"
        start = _start_of(text, "bar")
        assert prev_token(text, start) == "foo"

    def test_skips_whitespace_variants(self):
        text = "alpha\t  \n  beta"
        start = _start_of(text, "beta")
        assert prev_token(text, start) == "alpha"

    def test_includes_dots_in_token(self):
        text = "See v1.2.3 now"
        start = _start_of(text, "now")
        assert prev_token(text, start) == "v1.2.3"

    def test_includes_colon_in_token(self):
        text = "Arrive 12:30 soon"
        start = _start_of(text, "soon")
        assert prev_token(text, start) == "12:30"

    def test_hyphen_and_comma_break_tokens(self):
        text = "GPU-CPU next, okay"

        # Hyphen is NOT allowed, so previous token before "next" is the alnum run "CPU".
        start = _start_of(text, "next")
        assert prev_token(text, start) == "CPU"

        # Comma is a hard boundary and is *not* skipped by prev_token → returns ''.
        start2 = _start_of(text, "okay")
        assert prev_token(text, start2) == ""

        # If the start point is a comma, prev_token will recover "next".
        comma_idx = text.index(",")
        assert prev_token(text, comma_idx) == "next"

    def test_unicode_letters_are_alnum(self):
        text = "αλφα beta"
        start = _start_of(text, "beta")
        assert prev_token(text, start) == "αλφα"

    def test_start_at_zero_returns_empty(self):
        text = "anything"
        assert prev_token(text, 0) == ""

    def test_start_inside_current_token_returns_prefix_before_start(self):
        text = "foo bar baz"
        # point to 'a' inside "bar" → prev chunk before start is just "b"
        inside = _start_of(text, "bar") + 1
        assert prev_token(text, inside) == "b"

    def test_skips_whitespace(self):
        text = "alpha \t \n beta"
        assert prev_token(text, _start_of(text, "beta")) == "alpha"

    def test_includes_dot_and_colon_in_token(self):
        text = "See v1.2.3 at 12:30 now"
        assert prev_token(text, _start_of(text, "now")) == "12:30"
        assert prev_token(text, _start_of(text, "at")) == "v1.2.3"

    def test_unicode_letters(self):
        text = "αλφα beta"
        assert prev_token(text, _start_of(text, "beta")) == "αλφα"


class TestCoreLenForBounds:
    @pytest.mark.parametrize(
        "token,expected",
        [
            ("ABC", 3),
            ("A-B_C", 3),  # separators ignored
            ("R&D", 2),  # '&' ignored
            ("H2O", 3),  # digits counted
            ("U.S.A.", 3),  # dots ignored
            ("  A  ", 1),  # spaces ignored
            ("", 0),
            ("--//..", 0),  # only punctuation
            ("éß", 2),  # Unicode letters counted
            ("中A3!", 3),  # CJK + Latin + digit, '!' ignored
        ],
    )
    def test_counts_alnum_only(self, token, expected):
        assert core_len_for_bounds(token) == expected


class TestThresholdLen:
    def test_no_separators_returns_core_len(self, monkeypatch):
        # core_len_for_bounds is returned as-is when no allowed seps or dots
        monkeypatch.setattr(core, "core_len_for_bounds", lambda s: 5)
        assert threshold_len("GPUCPU", allow_chars="&/-") == 5

    def test_allowed_separator_upgrades_min_to_three_when_core_len_lt_3(self, monkeypatch):
        monkeypatch.setattr(core, "core_len_for_bounds", lambda s: 2)
        # '&' is in allow_chars → max(3, 2) == 3
        assert threshold_len("R&D", allow_chars="&/-") == 3

    def test_allowed_separator_with_core_len_ge_3_keeps_core_len(self, monkeypatch):
        monkeypatch.setattr(core, "core_len_for_bounds", lambda s: 6)
        # '-' is allowed but core_len already >= 3 → unchanged
        assert threshold_len("GPU-CPU", allow_chars="&/-") == 6

    def test_dot_triggers_upgrade_even_if_not_in_allow_chars(self, monkeypatch):
        monkeypatch.setattr(core, "core_len_for_bounds", lambda s: 2)
        # '.' not in allow_chars, but special-cased → max(3, 2) == 3
        assert threshold_len("U.S.", allow_chars="&/-") == 3

    def test_separator_not_in_allow_chars_does_not_upgrade(self, monkeypatch):
        monkeypatch.setattr(core, "core_len_for_bounds", lambda s: 2)
        # '&' NOT allowed here, and no '.' → stays 2
        assert threshold_len("R&D", allow_chars="-/") == 2

    def test_empty_allow_chars_behaviour(self, monkeypatch):
        monkeypatch.setattr(core, "core_len_for_bounds", lambda s: 4)
        # No allowed seps; no '.' → returns core len
        assert threshold_len("ABCD", allow_chars="") == 4

    def test_integration_no_patch_matches_core_len_when_plain_token(self):
        # With the real core_len_for_bounds, plain tokens should match exactly
        tok = "PlainToken"
        expected = core_len_for_bounds(tok)
        assert threshold_len(tok, allow_chars="&/-") == expected

class TestContainedInAny:
    @pytest.mark.parametrize(
        "s,e,containers,expected",
        [
            # Exact containment
            (12, 18, [("a", 10, 20)], True),
            # Exact equality on boundaries
            (10, 20, [("a", 10, 20)], True),
            # Multiple containers: later one contains
            (12, 18, [("a", 0, 5), ("b", 10, 20), ("c", 100, 200)], True),
            # Not contained: only overlaps on left edge
            (5, 10, [("a", 10, 20)], False),
            # Not contained: only touches on right edge
            (20, 25, [("a", 10, 20)], False),
            # Not contained: disjoint entirely
            (6, 9, [("a", 0, 5), ("b", 10, 15)], False),
            # Empty containers
            (1, 2, [], False),
        ],
    )
    def test_basic_cases(self, s, e, containers, expected):
        assert _contained_in_any(s, e, containers) is expected

    def test_early_exit_behavior_with_sorted_input(self):
        # Sorted by start: once ds > e, we can safely break and return False.
        s, e = 8, 9
        containers = [("a", 0, 3), ("b", 5, 7), ("c", 10, 15)]  # sorted by ds
        assert _contained_in_any(s, e, containers) is False

    def test_requires_sorted_by_start_for_correctness(self):
        # If containers are NOT sorted by start, early-exit can produce a false negative.
        s, e = 20, 30
        unsorted_containers = [
            ("early_big_start", 50, 100),  # ds > e triggers early break
            ("would_have_contained", 0, 1000),  # this actually contains (20,30)
        ]
        # Demonstrate current behavior (False) and document the contract expectation.
        assert _contained_in_any(s, e, unsorted_containers) is False
        # Recommended: callers must sort by ds ascending before calling:
        sorted_containers = sorted(unsorted_containers, key=lambda t: t[1])
        assert _contained_in_any(s, e, sorted_containers) is True

