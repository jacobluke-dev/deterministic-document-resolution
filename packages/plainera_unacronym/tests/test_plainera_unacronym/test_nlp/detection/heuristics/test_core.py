import re
import types

import plainera_unacronym.nlp.detection.detector as det
import plainera_unacronym.nlp.detection.heuristics.core as core
import plainera_unacronym.nlp.plugins.registry as domain_mod
import pytest
from plainera_unacronym.nlp import DetectorConfig
from plainera_unacronym.nlp.common.constants import TRAILING_PUNCT_CHARS
from plainera_unacronym.nlp.detection.heuristics.core import (
    _collect_core_hits,
    _collect_domain_hits,
    _contained_in_any,
    _has_lower_and_upper,
    caps_ratio,
    context_window,
    core_len_for_bounds,
    has_letter,
    has_stands_for_follow,
    in_brackets,
    iter_candidates_with,
    letters,
    next_word_lowercase,
    prev_token,
    strip_trailing_punct,
)


def _idx(text: str, token: str) -> tuple[int, int]:
    s = text.index(token)
    return s, s + len(token)


def _start_of(text: str, token: str) -> int:
    return text.index(token)


def _end(text: str, token: str) -> int:
    s = text.index(token)
    return s + len(token)


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

    def test_no_letters_returns_1(self):
        assert caps_ratio("") == 1.0
        assert caps_ratio("1234-._") == 1.0

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
        ns, ne = strip_trailing_punct(text, s, e)
        assert (ns, ne) == (s, e)
        assert text[ns:ne] == "GPU"

    def test_strip_single_period(self):
        text = "Memory uses RAM."
        # include '.' in the span
        s = text.index("RAM")
        e = s + len("RAM.")
        ns, ne = strip_trailing_punct(text, s, e)
        assert text[ns:ne] == "RAM"
        # the '.' should now be just outside the slice
        assert text[ne:ne + 1] == "."

    def test_strip_multiple_closers_chain(self):
        text = "Token!?) next"
        # span includes all three trailing punct chars
        s = text.index("Token")
        e = s + len("Token!?)")
        ns, ne = strip_trailing_punct(text, s, e)
        # all trailing punct removed, leaving bare token
        assert text[ns:ne] == "Token"

    def test_only_punctspan_becomes_empty(self):
        text = "Hello !!! there"
        s = text.index("!!!")
        e = s + 3
        ns, ne = strip_trailing_punct(text, s, e)
        assert ns == ne  # empty slice after stripping

    def test_parametric_known_trailing_chars(self):
        # Verify behavior for whatever is actually configured in TRAILING_PUNCT_DEFAULT
        base = "ACRONYM"
        for ch in [".", "!", "?", ")", "]", "'", '"', "”", ",", ";", ":"]:
            text = base + ch + " tail"
            s = 0
            e = len(base) + 1  # include the trailing char
            ns, ne = strip_trailing_punct(text, s, e)
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

        # If you point start at the comma, prev_token will recover "next".
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


class TestHasLetter:
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
        assert core.threshold_len("GPUCPU", allow_chars="&/-") == 5

    def test_allowed_separator_upgrades_min_to_three_when_core_len_lt_3(self, monkeypatch):
        monkeypatch.setattr(core, "core_len_for_bounds", lambda s: 2)
        # '&' is in allow_chars → max(3, 2) == 3
        assert core.threshold_len("R&D", allow_chars="&/-") == 3

    def test_allowed_separator_with_core_len_ge_3_keeps_core_len(self, monkeypatch):
        monkeypatch.setattr(core, "core_len_for_bounds", lambda s: 6)
        # '-' is allowed but core_len already >= 3 → unchanged
        assert core.threshold_len("GPU-CPU", allow_chars="&/-") == 6

    def test_dot_triggers_upgrade_even_if_not_in_allow_chars(self, monkeypatch):
        monkeypatch.setattr(core, "core_len_for_bounds", lambda s: 2)
        # '.' not in allow_chars, but special-cased → max(3, 2) == 3
        assert core.threshold_len("U.S.", allow_chars="&/-") == 3

    def test_separator_not_in_allow_chars_does_not_upgrade(self, monkeypatch):
        monkeypatch.setattr(core, "core_len_for_bounds", lambda s: 2)
        # '&' NOT allowed here, and no '.' → stays 2
        assert core.threshold_len("R&D", allow_chars="-/") == 2

    def test_empty_allow_chars_behaviour(self, monkeypatch):
        monkeypatch.setattr(core, "core_len_for_bounds", lambda s: 4)
        # No allowed seps; no '.' → returns core len
        assert core.threshold_len("ABCD", allow_chars="") == 4

    def test_integration_no_patch_matches_core_len_when_plain_token(self):
        # With the real core_len_for_bounds, plain tokens should match exactly
        tok = "PlainToken"
        expected = core.core_len_for_bounds(tok)
        assert core.threshold_len(tok, allow_chars="&/-") == expected


class TestBoostConfidenceIfWhitelisted:
    def _cfg(self, **overrides):
        # Minimal, flexible config object
        base = {
            "whitelist_two_letter": {"AI", "UK"},
            "two_letter_boost": 0.75,
            "dotted_display": "strip",
            "allow_chars": "&-/."
        }
        base.update(overrides)
        return types.SimpleNamespace(**base)

    def test_boosts_when_two_letter_and_whitelisted(self, monkeypatch):
        # Force the normalized key to 'AI'
        monkeypatch.setattr(core, "normalize_acronym_key",
                            lambda surface, **_: "AI",
                            raising=True)

        cfg = self._cfg()
        result = core.boost_confidence_if_whitelisted("A.I.", 0.20, cfg)
        assert result == pytest.approx(0.95)  # 0.20 + 0.75

    def test_caps_at_point_99(self, monkeypatch):
        monkeypatch.setattr(core, "normalize_acronym_key",
                            lambda surface, **_: "AI",
                            raising=True)

        cfg = self._cfg(two_letter_boost=0.75)
        result = core.boost_confidence_if_whitelisted("AI", 0.50, cfg)
        assert result == pytest.approx(0.99)  # capped

    def test_no_boost_when_not_whitelisted(self, monkeypatch):
        monkeypatch.setattr(core, "normalize_acronym_key",
                            lambda surface, **_: "TV",  # not in whitelist
                            raising=True)

        cfg = self._cfg()
        result = core.boost_confidence_if_whitelisted("TV", 0.40, cfg)
        assert result == pytest.approx(0.40)

    def test_no_boost_when_not_two_letters(self, monkeypatch):
        # Even if present in whitelist, length != 2 should not boost
        monkeypatch.setattr(core, "normalize_acronym_key",
                            lambda surface, **_: "GPU",
                            raising=True)

        cfg = self._cfg(whitelist_two_letter={"GPU"})  # irrelevant; len != 2
        result = core.boost_confidence_if_whitelisted("GPU", 0.33, cfg)
        assert result == pytest.approx(0.33)

    def test_respects_custom_boost_from_cfg(self, monkeypatch):
        monkeypatch.setattr(core, "normalize_acronym_key",
                            lambda surface, **_: "UK",
                            raising=True)

        cfg = self._cfg(two_letter_boost=0.10)
        result = core.boost_confidence_if_whitelisted("U.K.", 0.50, cfg)
        assert result == pytest.approx(0.60)

    def test_uses_defaults_when_cfg_lacks_optional_attrs(self, monkeypatch):
        # Capture kwargs to ensure defaults (allow_chars, dotted_mode) are passed
        seen = {}
        def _fake_normalize(surface, **kwargs):
            seen.update(kwargs)
            return "AI"

        monkeypatch.setattr(core, "normalize_acronym_key", _fake_normalize, raising=True)

        # cfg without dotted_display / allow_chars / two_letter_boost
        cfg = types.SimpleNamespace(whitelist_two_letter={"AI"})
        result = core.boost_confidence_if_whitelisted("A.I.", 0.10, cfg)

        # Default boost = 0.75 ⇒ 0.85
        assert result == pytest.approx(0.85)
        # Function should fall back to defaults inside getattr calls
        assert seen.get("allow_chars") == "&-/."
        assert seen.get("dotted_mode") == "strip"


class TestScoreUnit:
    def test_base_score_no_signals(self, monkeypatch):
        # in_brackets -> (False, False), no paren def, no "stands for"
        monkeypatch.setattr(core, "in_brackets", lambda t, s, e: (False, False))
        monkeypatch.setattr(det, "has_paren_definition", lambda t, e: False, raising=False)
        monkeypatch.setattr(det, "has_stands_for_follow", lambda t, e: False, raising=False)

        cfg = DetectorConfig()
        text = "We use GPU daily."
        s, e = _idx(text, "GPU")
        assert det.score("GPU", text, s, e, cfg) == 0.6

    def test_in_brackets_inside_adds_point_25(self, monkeypatch):
        monkeypatch.setattr(core, "in_brackets", lambda t, s, e: (True, False))
        monkeypatch.setattr(det, "has_paren_definition", lambda t, e: False, raising=False)
        monkeypatch.setattr(det, "has_stands_for_follow", lambda t, e: False, raising=False)

        cfg = DetectorConfig()
        text = "(GPU) is fast."
        s, e = _idx(text, "GPU")
        assert det.score("GPU", text, s, e, cfg) == 0.6 + 0.25

    def test_inside_takes_precedence_over_adjacent(self, monkeypatch):
        # If both are True, only inside (+0.25) applies due to elif
        monkeypatch.setattr(core, "in_brackets", lambda t, s, e: (True, True))
        monkeypatch.setattr(det, "has_paren_definition", lambda t, e: False, raising=False)
        monkeypatch.setattr(det, "has_stands_for_follow", lambda t, e: False, raising=False)

        cfg = DetectorConfig()
        text = "GPU near brackets."
        s, e = _idx(text, "GPU")
        assert det.score("GPU", text, s, e, cfg) == 0.85

    def test_paren_definition_adds_point_25(self, monkeypatch):
        monkeypatch.setattr(core, "in_brackets", lambda t, s, e: (False, False))
        monkeypatch.setattr(det, "has_paren_definition", lambda t, e: True, raising=False)
        monkeypatch.setattr(det, "has_stands_for_follow", lambda t, e: False, raising=False)

        cfg = DetectorConfig()
        text = "GPU (Graphics Processing Unit)"
        s, e = _idx(text, "GPU")
        assert det.score("GPU", text, s, e, cfg) == 0.6 + 0.25

    def test_stands_for_follow_adds_point_15(self, monkeypatch):
        monkeypatch.setattr(core, "in_brackets", lambda t, s, e: (False, False))
        monkeypatch.setattr(det, "has_paren_definition", lambda t, e: False, raising=False)
        monkeypatch.setattr(det, "has_stands_for_follow", lambda t, e: True, raising=False)

        cfg = DetectorConfig()
        text = "GPU stands for Graphics Processing Unit."
        s, e = _idx(text, "GPU")
        assert det.score("GPU", text, s, e, cfg) == 0.6 + 0.15

    def test_soft_blacklist_penalises_point_2(self, monkeypatch):
        monkeypatch.setattr(core, "in_brackets", lambda t, s, e: (False, False))
        monkeypatch.setattr(det, "has_paren_definition", lambda t, e: False, raising=False)
        monkeypatch.setattr(det, "has_stands_for_follow", lambda t, e: False, raising=False)

        cfg = DetectorConfig()
        text = "We saw AS today."
        s, e = _idx(text, "AS")
        # AS is in cfg.soft_blacklist → -0.2
        assert det.score("AS", text, s, e, cfg) == 0.6 - 0.2

    def test_upper_bound_clamped_to_one(self, monkeypatch):
        # Base 0.6 + inside .25 + paren .25 + stands_for .15 = 1.25 → clamp to 1.0
        monkeypatch.setattr(core, "in_brackets", lambda t, s, e: (True, False))
        monkeypatch.setattr(det, "has_paren_definition", lambda t, e: True, raising=False)
        monkeypatch.setattr(det, "has_stands_for_follow", lambda t, e: True, raising=False)

        cfg = DetectorConfig()
        text = "GPU (Graphics Processing Unit). GPU stands for Graphics Processing Unit."
        s, e = _idx(text, "GPU")
        assert det.score("GPU", text, s, e, cfg) == 1.0


class TestScoreIntegration:
    def test_score_with_real_heuristics_stands_for(self):
        # No patching: rely on real implementations.
        # Pattern: "<TOKEN> stands for <definition>" should add +0.15.
        text = "In docs, GPU stands for Graphics Processing Unit."
        s, e = _idx(text, "GPU")
        val = det.score("GPU", text, s, e, DetectorConfig())

        # Expect base 0.6 + 0.15 for 'stands for', possibly more if your in_brackets
        # logic treats proximity to punctuation as adjacent—but there are no brackets here.
        assert val == 0.75


class TestContextWindow:
    def test_snaps_to_sentence_bounds_and_includes_terminator(self):
        text = "Alpha bravo. Charlie delta echo! Foxtrot golf?"
        start, end = _idx(text, "delta")
        left, right = context_window(text, start, end, window_chars=1_000)
        # Left snaps to first non-space after previous terminator ('. ')
        assert text[left:right] == "Charlie delta echo!"
        assert text[left] == "C"
        assert text[right - 1] == "!"

    def test_skips_left_whitespace_after_newline(self):
        text = "One.\n   Two things here.\nThird."
        start, end = _idx(text, "Two")
        left, right = context_window(text, start, end, window_chars=999)
        # Left should skip '\n' + spaces; effectively equals the start of "Two"
        assert left == start
        assert text[left:right].startswith("Two")
        assert text[right - 1] == "."

    def test_small_window_limits_left(self):
        text = "Hello amazing world."
        start, end = _idx(text, "amazing")
        left, right = context_window(text, start, end, window_chars=3)
        # When limited by window_chars, we don't snap to boundary; we stop mid-sentence
        assert left == start - 3
        assert right >= end  # right still moves independently
        # Slice should *not* start at word boundary necessarily
        assert text[left:right].startswith(text[start - 3:start])

    def test_small_window_limits_right(self):
        text = "Hello amazing world."
        start, end = _idx(text, "amazing")
        left, right = context_window(text, start, end, window_chars=2)
        # Right should advance only 2 chars beyond end (no terminator bump)
        assert right == end + 2
        assert text[right - 1] != "."  # did not include sentence terminator

    def test_no_terminators_expands_to_edges(self):
        text = "no punctuation anywhere at all"
        start, end = _idx(text, "punctuation")
        left, right = context_window(text, start, end, window_chars=10_000)
        assert left == 0
        assert right == len(text)
        assert text[left:right] == text

    def test_right_immediate_terminator_is_included(self):
        text = "Token."
        start, end = _idx(text, "Token")
        left, right = context_window(text, start, end, window_chars=100)
        # Right should include the '.' terminator
        assert text[right - 1] == "."
        assert text[left:right] == "Token."

    @pytest.mark.parametrize("sep", ["\n", "\r"])
    def test_newline_and_cr_act_as_terminators(self, sep: str):
        text = f"First line{sep}Second line."
        start, end = _idx(text, "Second")
        left, right = context_window(text, start, end, window_chars=999)
        # Left snaps to just after the newline/CR
        assert left == start
        assert text[left:right].startswith("Second")
        assert text[right - 1] == "."

    @pytest.mark.parametrize(
        "text,token,window",
        [
            ("Short end", "end", 5),
            ("A sentence with words.", "with", 4),
            ("Edge\ncase here.", "case", 3),
        ],
    )
    def test_bounds_are_well_formed(self, text, token, window):
        start, end = _idx(text, token)
        left, right = context_window(text, start, end, window_chars=window)
        assert 0 <= left <= start <= end <= right <= len(text)


class TestHasLowerAndUpper:
    @pytest.mark.parametrize(
        "tok,expected",
        [
            ("ABC", False),  # all upper
            ("abc", False),  # all lower
            ("aBc", True),  # mixed ASCII
            ("iOS", True),  # lower + upper
            ("H2O", False),  # digits ignored; no lowercase letters
            ("NaCl", True),  # camelcase style
            ("", False),  # empty
            ("123", False),  # digits only
            ("__-__", False),  # punctuation only
            ("a-A", True),  # symbols ignored; still both lower+upper
        ],
    )
    def test_ascii_and_symbols(self, tok, expected):
        assert _has_lower_and_upper(tok) is expected

    @pytest.mark.parametrize(
        "tok,expected",
        [
            ("Éclair", True),  # accented upper + lowercase
            ("ßA", True),  # German sharp s (lower) + uppercase
            ("İi", True),  # Turkish capital dotted I + lowercase i
            ("Δx", True),  # Greek uppercase + Latin lowercase
            ("中A", False),  # CJK isalpha=True but caseless; no lowercase present
            ("中aA", True),  # caseless + lower + upper → True
        ],
    )
    def test_unicode_cases(self, tok, expected):
        assert _has_lower_and_upper(tok) is expected


class DummyCfg(DetectorConfig):
    def __init__(
        self,
        min_len=2,
        max_len=10,
        require_caps_ratio=0.8,
        enable_mixed_case=False,
        require_caps_ratio_mixed=0.6,
        enabled_domains=frozenset(),
    ):
        object.__setattr__(self, "min_len", min_len)
        object.__setattr__(self, "max_len", max_len)
        object.__setattr__(self, "require_caps_ratio", require_caps_ratio)
        object.__setattr__(self, "enable_mixed_case", enable_mixed_case)
        object.__setattr__(self, "require_caps_ratio_mixed", require_caps_ratio_mixed)
        object.__setattr__(self, "enabled_domains", enabled_domains)


class TestAcceptCandidate:
    def test_trailing_punct_is_stripped_and_returns_span(self, monkeypatch):
        text = "ABC!"
        cfg = DummyCfg(min_len=3, max_len=10, require_caps_ratio=0.8)

        # strip trailing '!' -> (0,3)
        monkeypatch.setattr(core, "strip_trailing_punct", lambda t, s, e: (s, e - 1), raising=False)
        monkeypatch.setattr(core, "has_letter", lambda s: True, raising=False)
        monkeypatch.setattr(core, "core_len_for_bounds", lambda s: len(s), raising=False)
        monkeypatch.setattr(core, "caps_ratio", lambda s: 1.0, raising=False)
        monkeypatch.setattr(core, "_has_lower_and_upper", lambda s: False, raising=False)

        out = core._accept_candidate(text, cfg, 0, len(text))
        assert out == ("ABC", 0, 3)

    def test_rejects_when_no_letters(self, monkeypatch):
        text = "123-456"
        cfg = DummyCfg(min_len=2)

        monkeypatch.setattr(core, "strip_trailing_punct", lambda t, s, e: (s, e), raising=False)
        monkeypatch.setattr(core, "has_letter", lambda s: False, raising=False)

        # The rest shouldn't matter if has_letter is False, but provide safe defaults
        monkeypatch.setattr(core, "core_len_for_bounds", lambda s: 5, raising=False)
        monkeypatch.setattr(core, "caps_ratio", lambda s: 1.0, raising=False)
        monkeypatch.setattr(core, "_has_lower_and_upper", lambda s: False, raising=False)

        assert core._accept_candidate(text, cfg, 0, len(text)) is None

    @pytest.mark.parametrize(
        "core_len,min_len,max_len,expected_none",
        [
            (1, 1, 10, True),  # explicit guard: core_len == 1 -> reject
            (1, 2, 10, True),  # below min_len by core_len check
            (16, 2, 20, True),  # explicit guard: core_len >= 15 -> reject
            (12, 2, 10, True),  # above max_len
            (5, 2, 10, False),  # within bounds -> not rejected by length gate
        ],
    )
    def test_core_length_guards(self, monkeypatch, core_len, min_len, max_len, expected_none):
        text = "AAAAAAAAAAAAAAAAAAAA"  # long enough so first raw len check passes
        s, e = 0, 10  # e - s = 10 >= min_len usually
        cfg = DummyCfg(min_len=min_len, max_len=max_len)

        monkeypatch.setattr(core, "strip_trailing_punct", lambda t, _s, _e: (s, e), raising=False)
        monkeypatch.setattr(core, "has_letter", lambda srf: True, raising=False)
        monkeypatch.setattr(core, "core_len_for_bounds", lambda srf: core_len, raising=False)
        monkeypatch.setattr(core, "caps_ratio", lambda srf: 1.0, raising=False)
        monkeypatch.setattr(core, "_has_lower_and_upper", lambda srf: False, raising=False)

        out = core._accept_candidate(text, cfg, s, e)
        assert (out is None) == expected_none

    @pytest.mark.parametrize(
        "ratio,threshold,accepted",
        [
            (0.79, 0.80, False),
            (0.80, 0.80, True),
            (1.00, 0.80, True),
        ],
    )
    def test_caps_ratio_threshold_without_mixed_case(self, monkeypatch, ratio, threshold, accepted):
        text = "AbCD"  # content irrelevant; we control caps_ratio directly
        s, e = 0, len(text)
        cfg = DummyCfg(min_len=2, max_len=10, require_caps_ratio=threshold, enable_mixed_case=False)

        monkeypatch.setattr(core, "strip_trailing_punct", lambda t, _s, _e: (s, e), raising=False)
        monkeypatch.setattr(core, "has_letter", lambda srf: True, raising=False)
        monkeypatch.setattr(core, "core_len_for_bounds", lambda srf: 4, raising=False)
        monkeypatch.setattr(core, "caps_ratio", lambda srf: ratio, raising=False)
        monkeypatch.setattr(core, "_has_lower_and_upper", lambda srf: False, raising=False)

        out = core._accept_candidate(text, cfg, s, e)
        assert (out is not None) == accepted

    def test_mixed_case_relaxation_applies_when_enabled_and_both_cases_and_two_uppers(self, monkeypatch):
        text = "AbC"  # has both lower and upper; uppers=2 ('A','C')
        s, e = 0, len(text)
        cfg = DummyCfg(
            min_len=2,
            max_len=10,
            require_caps_ratio=0.9,
            enable_mixed_case=True,
            require_caps_ratio_mixed=0.5,
        )

        monkeypatch.setattr(core, "strip_trailing_punct", lambda t, _s, _e: (s, e), raising=False)
        monkeypatch.setattr(core, "has_letter", lambda srf: True, raising=False)
        monkeypatch.setattr(core, "core_len_for_bounds", lambda srf: 3, raising=False)
        # Raw ratio is too low for 0.9 but above the mixed threshold 0.5
        monkeypatch.setattr(core, "caps_ratio", lambda srf: 0.6, raising=False)
        monkeypatch.setattr(core, "_has_lower_and_upper", lambda srf: True, raising=False)

        out = core._accept_candidate(text, cfg, s, e)
        assert out == (text, s, e)

    def test_no_relaxation_if_mixed_case_disabled_or_fewer_than_two_uppers(self, monkeypatch):
        # Case A: mixed-case disabled -> requires 0.9 and should fail at 0.6
        text = "Ab"  # only one upper
        s, e = 0, len(text)

        # Disabled: expect reject
        cfg_disabled = DummyCfg(
            min_len=2, max_len=10, require_caps_ratio=0.9, enable_mixed_case=False, require_caps_ratio_mixed=0.5
        )
        monkeypatch.setattr(core, "strip_trailing_punct", lambda t, _s, _e: (s, e), raising=False)
        monkeypatch.setattr(core, "has_letter", lambda srf: True, raising=False)
        monkeypatch.setattr(core, "core_len_for_bounds", lambda srf: 2, raising=False)
        monkeypatch.setattr(core, "caps_ratio", lambda srf: 0.6, raising=False)
        monkeypatch.setattr(core, "_has_lower_and_upper", lambda srf: True, raising=False)
        assert core._accept_candidate(text, cfg_disabled, s, e) is None

        # Enabled but <2 uppers -> still no relaxation -> reject
        cfg_enabled = DummyCfg(
            min_len=2, max_len=10, require_caps_ratio=0.9, enable_mixed_case=True, require_caps_ratio_mixed=0.5
        )
        assert core._accept_candidate(text, cfg_enabled, s, e) is None

    def test_min_len_raw_slice_check_happens_before_surface_processing(self, monkeypatch):
        # Raw slice length check: e - s < cfg.min_len => immediate reject
        text = "ABCDE"
        cfg = DummyCfg(min_len=6, max_len=10)

        calls = {"strip_called": False}

        def strip_fn(t, s, e):
            calls["strip_called"] = True
            return (s, e)

        # strip_trailing_punct is still called (your function calls it before the raw len check),
        # but the early length check will return None before any deeper gates.
        monkeypatch.setattr(core, "strip_trailing_punct", strip_fn, raising=False)
        monkeypatch.setattr(core, "has_letter", lambda s: True, raising=False)
        monkeypatch.setattr(core, "core_len_for_bounds", lambda s: 5, raising=False)
        monkeypatch.setattr(core, "caps_ratio", lambda s: 1.0, raising=False)
        monkeypatch.setattr(core, "_has_lower_and_upper", lambda s: False, raising=False)

        out = core._accept_candidate(text, cfg, 0, 5)
        assert out is None
        assert calls["strip_called"] is True

    def _span(self, text: str, token: str) -> tuple[int, int]:
        s = text.index(token)
        return s, s + len(token)

    def test_end_to_end_accept_reject_matrix(self):
        # Configure with realistic bounds and mixed-case relaxation.
        cfg = DetectorConfig(
            min_len=2,
            max_len=10,
            require_caps_ratio=0.80,
            enable_mixed_case=True,
            require_caps_ratio_mixed=0.50,
        )

        # NOTE: We’ll take spans directly over the text and feed them in.
        text = "GPU, A WiFi NASA A THISISMORETHANFIFTEEN 1234 AbC!"
        # Cases:
        # - "GPU,"   -> accept; should strip trailing comma, return ("GPU", s, e_without_comma)
        # - "A"      -> reject: explicit core_len==1 guard
        # - "WiFi"   -> accept due to mixed-case relaxation (2 uppers, ratio≈0.5, meets mixed 0.5)
        # - "NASA"   -> accept (all caps, length within bounds)
        # - "THISISMORETHANFIFTEEN" -> reject: core_len >= 15 guard
        # - "1234"   -> reject: no letters
        # - "AbC!"   -> accept due to mixed-case relaxation; strips '!' at end

        # Accept: GPU,
        s, e = self._span(text, "GPU,")
        out_gpu = core._accept_candidate(text, cfg, s, e)
        assert out_gpu == ("GPU", s, s + 3)  # comma stripped

        # Reject: single-char "A"
        s, e = self._span(text, " A ")
        s += 1  # point to the 'A'
        out_a = core._accept_candidate(text, cfg, s, s + 1)
        assert out_a is None

        # Accept: WiFi (mixed-case relaxation applies: uppers >= 2, ratio ~ 0.5)
        s, e = self._span(text, "WiFi")
        out_wifi = core._accept_candidate(text, cfg, s, e)
        assert out_wifi == ("WiFi", s, e)

        # Accept: NASA (straight all-caps)
        s, e = self._span(text, "NASA")
        out_nasa = core._accept_candidate(text, cfg, s, e)
        assert out_nasa == ("NASA", s, e)

        # Reject: >=15 letters
        s, e = self._span(text, "THISISMORETHANFIFTEEN")
        out_long = core._accept_candidate(text, cfg, s, e)
        assert out_long is None

        # Reject: no letters
        s, e = self._span(text, "1234")
        out_digits = core._accept_candidate(text, cfg, s, e)
        assert out_digits is None

        # Accept: AbC! (mixed-case + trailing punct strip)
        s, e = self._span(text, "AbC!")
        out_abc = core._accept_candidate(text, cfg, s, e)
        assert out_abc == ("AbC", s, s + 3)  # '!' stripped


class TestCollectCoreHits:

    def test_collects_in_text_order(self, monkeypatch):
        # Pattern: named group 'tok' for ALL-CAPS tokens length>=2
        pat = re.compile(r"(?P<tok>[A-Z]{2,})")

        text = "xx ABC yy DEF and GHIJ."
        # Echo back a Span-like tuple to simulate acceptance
        monkeypatch.setattr(core, "_accept_candidate", lambda _t, _c, s, e: ("hit", s, e), raising=False)

        # Minimal config object (fields unused by our stub)
        class DetectorConfig: ...

        cfg = DetectorConfig()

        hits = _collect_core_hits(text, cfg, pat)
        # Expect left-to-right order by match positions
        assert hits == [
            ("hit", text.index("ABC"), text.index("ABC") + 3),
            ("hit", text.index("DEF"), text.index("DEF") + 3),
            ("hit", text.index("GHIJ"), text.index("GHIJ") + 4),
        ]

    def test_rejected_hits_are_filtered_out(self, monkeypatch):
        pat = re.compile(r"(?P<tok>[A-Z]{2,})")
        text = "ABC DEF GHI"

        def accept(_t, _c, s, e):
            # Reject the middle token DEF
            tok = text[s:e]
            return None if tok == "DEF" else ("hit", s, e)

        monkeypatch.setattr(core, "_accept_candidate", accept, raising=False)

        class DetectorConfig: ...

        cfg = DetectorConfig()

        hits = _collect_core_hits(text, cfg, pat)
        assert [text[s:e] for (_, s, e) in hits] == ["ABC", "GHI"]

    def test_no_matches_returns_empty(self, monkeypatch):
        pat = re.compile(r"(?P<tok>[A-Z]{2,})")
        text = "no caps here"

        monkeypatch.setattr(core, "_accept_candidate", lambda *_: ("hit", 0, 0), raising=False)

        class DetectorConfig: ...

        cfg = DetectorConfig()

        assert _collect_core_hits(text, cfg, pat) == []

    def test_requires_named_group_tok(self, monkeypatch):
        # Pattern without 'tok' should raise (m.span('tok') IndexError)
        pat = re.compile(r"([A-Z]{2,})")
        text = "ABC"

        monkeypatch.setattr(core, "_accept_candidate", lambda *_: ("hit", 0, 3), raising=False)

        class DetectorConfig: ...

        cfg = DetectorConfig()

        with pytest.raises(IndexError):
            _collect_core_hits(text, cfg, pat)


class TestCollectDomainHits:
    def test_collects_and_sorts_by_start_then_length_desc(self, monkeypatch):
        text = "The ABC transporter and ABCDEF domain are here."

        # Fake plugins
        class BioPlug:
            def extra_candidates(self, _text, _cfg):
                # Same start 10, different lengths; plus another span at start 5
                return [
                    ("bio", 10, 15),  # len 5
                    ("bio", 10, 20),  # len 10 (should come before len 5)
                    ("bio", 5, 7),  # earlier start (should be first overall)
                ]

        class FinPlug:
            def extra_candidates(self, _text, _cfg):
                return [("fin", 12, 18)]  # middle

        monkeypatch.setattr(
            core, "DOMAIN_PLUGINS", {"bio": BioPlug(), "finance": FinPlug()}, raising=False
        )

        # Accept everything by echoing a Span-like tuple
        def accept(text_arg, cfg_arg, s, e):
            return ("hit", s, e)

        monkeypatch.setattr(core, "_accept_candidate", accept, raising=False)

        cfg = DetectorConfig(enabled_domains=("bio", "finance"))
        hits = _collect_domain_hits(text, cfg)

        # Expect sort: start asc (5..7), then 10..20 (longer first), then 10..15, then 12..18
        assert hits == [
            ("hit", 5, 7),
            ("hit", 10, 20),
            ("hit", 10, 15),
            ("hit", 12, 18),
        ]

    def test_filters_by_enabled_domains_only(self, monkeypatch):
        class BioPlug:
            def extra_candidates(self, *_):
                return [("bio", 1, 3)]

        class FinPlug:
            def extra_candidates(self, *_):
                return [("fin", 100, 110)]

        monkeypatch.setattr(core, "DOMAIN_PLUGINS",
                            {"bio": BioPlug(), "finance": FinPlug()}, raising=False)
        monkeypatch.setattr(core, "_accept_candidate",
                            lambda _t, _c, s, e: ("hit", s, e), raising=False)

        cfg = DetectorConfig(enabled_domains=frozenset({"bio"}))  # finance disabled
        hits = _collect_domain_hits("x", cfg)
        assert hits == [("hit", 1, 3)]

    def test_skips_missing_plugins_and_rejections_and_none_returns(self, monkeypatch):
        class BioPlug:
            def extra_candidates(self, *_):
                return [("bio", 5, 9), ("bio", 50, 60)]

        monkeypatch.setattr(core, "DOMAIN_PLUGINS", {"bio": BioPlug()}, raising=False)

        def accept(_t, _c, s, e):
            return None if (s, e) == (50, 60) else ("hit", s, e)

        monkeypatch.setattr(core, "_accept_candidate", accept, raising=False)

        cfg = DetectorConfig(enabled_domains=frozenset({"bio", "chem"}))  # "chem" missing -> ignored
        hits = _collect_domain_hits("x", cfg)
        assert hits == [("hit", 5, 9)]

    def test_empty_enabled_domains_or_none_yields_no_hits(self, monkeypatch):
        # Even if there are plugins, with enabled_domains empty/None the loop is skipped.
        class AnyPlug:
            def extra_candidates(self, *_):
                return [("x", 1, 2)]

        monkeypatch.setattr(domain_mod, "DOMAIN_PLUGINS", {"any": AnyPlug()}, raising=False)
        monkeypatch.setattr(
            domain_mod, "_accept_candidate", lambda *_: ("hit", 1, 2), raising=False
        )

        assert _collect_domain_hits("x", DetectorConfig(enabled_domains=(frozenset()))) == []
        assert _collect_domain_hits("x", DetectorConfig(enabled_domains=None)) == []


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


class TestIterCandidatesWith:
    # Token pattern for tests:
    # - Captures a "token" as letters followed by letters/digits or common separators.
    # - Includes '.' so we can verify trailing-punct trimming (e.g., "RAM.")
    PAT = re.compile(r"(?P<tok>[A-Za-z][A-Za-z0-9&\-/\.]*)")

    @staticmethod
    def collect(text: str, cfg: DetectorConfig, pat: re.Pattern[str]):
        return list(iter_candidates_with(text, cfg, pat))

    def test_all_caps_simple(self):
        cfg = DetectorConfig()
        text = "We ran it on the GPU and CPU yesterday."
        out = self.collect(text, cfg, self.PAT)
        surfaces = [s for s, _, _ in out]
        assert "GPU" in surfaces
        assert "CPU" in surfaces

    def test_mixed_case_relaxation_enabled(self):
        # "iOS": 2/3 letters uppercase ≈ 0.667. With mixed-case relaxation (0.5) it should pass.
        cfg = DetectorConfig(enable_mixed_case=True, require_caps_ratio_mixed=0.5)
        text = "We ship an iOS build every week."
        out = self.collect(text, cfg, self.PAT)
        surfaces = [s for s, _, _ in out]
        assert "iOS" in surfaces  # relaxed threshold applied (upp >= 2)

    def test_mixed_case_relaxation_disabled(self):
        # With relaxation OFF, require_caps_ratio=0.7 and iOS has ~0.667 → should be filtered out.
        cfg = DetectorConfig(enable_mixed_case=False)
        text = "We ship an iOS build every week."
        out = self.collect(text, cfg, self.PAT)
        surfaces = [s for s, _, _ in out]
        assert "iOS" not in surfaces

    def test_digits_ignored_in_caps_ratio(self):
        # "H2O": letters H,O are uppercase; digit '2' ignored → ratio = 1.0 → passes.
        cfg = DetectorConfig()
        text = "Check the H2O level."
        out = self.collect(text, cfg, self.PAT)
        surfaces = [s for s, _, _ in out]
        assert "H2O" in surfaces

    def test_trailing_punct_stripped_and_indices(self):
        cfg = DetectorConfig()
        text = "Memory uses RAM."
        out = self.collect(text, cfg, self.PAT)
        # Expect one candidate "RAM" with indices pointing exactly to 'RAM' (not the '.')
        assert any(s == "RAM" for s, _, _ in out)
        # Verify indices are tight to the token (no trailing '.')
        for srf, s, e in out:
            if srf == "RAM":
                assert text[s:e] == "RAM"
                # e should be the position right after 'M'
                assert text[e: e + 1] == "."  # the '.' is outside the candidate

    def test_min_len_enforced(self):
        # Default min_len=2 → single-letter tokens should be filtered.
        cfg = DetectorConfig()
        text = "A B CD"
        out = self.collect(text, cfg, self.PAT)
        surfaces = [s for s, _, _ in out]
        assert "A" not in surfaces
        assert "B" not in surfaces
        assert "CD" in surfaces

    def test_max_len_enforced(self):
        # Default max_len=10 → very long all-caps should be filtered.
        cfg = DetectorConfig()
        long_tok = "THISISVERYLONG"  # length 14
        text = f"Edge {long_tok} token."
        out = self.collect(text, cfg, self.PAT)
        surfaces = [s for s, _, _ in out]
        assert long_tok not in surfaces

    def test_allowed_separators_compound_tokens(self):
        # Ensure tokens with separators (&, -) get considered and pass.
        cfg = DetectorConfig()
        text = "Our R&D team ported GPU-CPU pipelines."
        out = self.collect(text, cfg, self.PAT)
        surfaces = [s for s, _, _ in out]
        assert "R&D" in surfaces
        assert "GPU-CPU" in surfaces

    def test_mixed_case_requires_two_uppers_for_relax(self):
        # Relaxation only kicks in if upp >= 2. "eBay" has only 1 uppercase in practice (B),
        # so it should fail under default require_caps_ratio=0.7.
        cfg = DetectorConfig(enable_mixed_case=True)
        text = "We listed it on eBay."
        out = self.collect(text, cfg, self.PAT)
        surfaces = [s for s, _, _ in out]
        assert "eBay" not in surfaces

    def test_mixed_case_relax_threshold_param(self):
        # Tighten the mixed-case threshold so "NaCl" (2/4 = 0.5) fails.
        cfg = DetectorConfig(enable_mixed_case=True, require_caps_ratio_mixed=0.6)
        text = "We used NaCl in the experiment."
        out = self.collect(text, cfg, self.PAT)
        surfaces = [s for s, _, _ in out]
        assert "NaCl" not in surfaces
