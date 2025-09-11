import re
import pytest

from plainera_unacronym.nlp import DetectorConfig
from plainera_unacronym.nlp.heuristics.core import _has_lower_and_upper, iter_candidates_with, context_window, has_letter, core_len_for_bounds, normalize_key
import plainera_unacronym.nlp.detector as det
import plainera_unacronym.nlp.heuristics.core as core

def _idx(text: str, token: str) -> tuple[int, int]:
    s = text.index(token)
    return s, s + len(token)

# Adjust the import if prev_token lives in a different module.
from plainera_unacronym.nlp.heuristics.core import prev_token


def _start_of(text: str, token: str) -> int:
    return text.index(token)


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

        # Hyphen is NOT allowed, so scanning left from "next" stops at '-'
        # → previous token is just the alnum run "CPU".
        start = _start_of(text, "next")
        assert prev_token(text, start) == "CPU"

        # Comma is a hard boundary; scanning left from "okay" stops at ',' → "next".
        start2 = _start_of(text, "okay")
        assert prev_token(text, start2) == "next"

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


class TestNormalizeKey:
    def test_dotted_mode_strip(self):
        assert normalize_key("U.S.A.", allow_chars="&-/", dotted_mode="strip") == "USA"

    def test_dotted_mode_preserve(self):
        assert normalize_key("U.S.A.", allow_chars="&-/", dotted_mode="preserve") == "U.S.A."

    def test_swallow_spaces_ampersand(self):
        assert normalize_key("R & D", allow_chars="&-/", dotted_mode="strip") == "R&D"
        assert normalize_key("R& D", allow_chars="&-/", dotted_mode="strip") == "R&D"
        assert normalize_key("R &D", allow_chars="&-/", dotted_mode="strip") == "R&D"
        assert normalize_key("R&D", allow_chars="&-/", dotted_mode="strip") == "R&D"

    def test_swallow_spaces_hyphen(self):
        # Single spaces on either/both sides collapse correctly
        assert normalize_key("GPU - CPU", allow_chars="-&/", dotted_mode="preserve") == "GPU-CPU"
        assert normalize_key("GPU- CPU", allow_chars="-&/", dotted_mode="preserve") == "GPU-CPU"
        assert normalize_key("GPU -CPU", allow_chars="-&/", dotted_mode="preserve") == "GPU-CPU"
        assert normalize_key("GPU-CPU", allow_chars="-&/", dotted_mode="preserve") == "GPU-CPU"

    def test_swallow_spaces_slash(self):
        assert normalize_key("A / B", allow_chars="/", dotted_mode="strip") == "A/B"
        assert normalize_key("A/ B", allow_chars="/", dotted_mode="strip") == "A/B"
        assert normalize_key("A /B", allow_chars="/", dotted_mode="strip") == "A/B"
        assert normalize_key("A/B", allow_chars="/", dotted_mode="strip") == "A/B"

    def test_non_allowed_separator_keeps_spaces(self):
        # '&' is not allowed here → spaces remain
        assert normalize_key("R & D", allow_chars="-/", dotted_mode="strip") == "R & D"

    def test_apostrophe_variants_are_canonicalized(self):
        # Curly apostrophe should normalize to ASCII "'"
        assert normalize_key("O’Reilly", allow_chars="&-/", dotted_mode="preserve") == "O'Reilly"

    def test_dash_variants_are_canonicalized_and_trimmed(self):
        # EN dash / EM dash should map to '-' then spacing rule applies
        assert normalize_key("GPU – CPU", allow_chars="-", dotted_mode="preserve") == "GPU-CPU"
        assert normalize_key("A—B", allow_chars="-", dotted_mode="preserve") == "A-B"

    def test_mixed_multiple_allowed_separators(self):
        s = "R & D / E"
        out = normalize_key(s, allow_chars="&/", dotted_mode="strip")
        assert out == "R&D/E"

    def test_allowed_at_edges(self):
        # Leading/trailing spaces around an allowed separator are swallowed appropriately
        assert normalize_key("A &B", allow_chars="&", dotted_mode="preserve") == "A&B"
        assert normalize_key("A& B", allow_chars="&", dotted_mode="preserve") == "A&B"


class TestHasLetter:
    @pytest.mark.parametrize(
        "s,expected",
        [
            ("abc", True),
            ("ABC", True),
            ("a1!", True),          # mixed, has a letter
            ("", False),
            ("123", False),         # digits only
            ("!!!", False),         # punctuation only
            (" \t\n", False),       # whitespace only
            ("   A   ", True),      # letters among spaces
            ("é", True),            # accented letter
            ("ß", True),            # Unicode letter
            ("Δ", True),            # Greek letter
            ("中", True),           # CJK letter
            ("🙂", False),          # emoji is not alpha
        ],
    )
    def test_various_strings(self, s, expected):
        assert has_letter(s) is expected


class TestCoreLenForBounds:
    @pytest.mark.parametrize(
        "token,expected",
        [
            ("ABC", 3),
            ("A-B_C", 3),           # separators ignored
            ("R&D", 2),             # '&' ignored
            ("H2O", 3),             # digits counted
            ("U.S.A.", 3),          # dots ignored
            ("  A  ", 1),           # spaces ignored
            ("", 0),
            ("--//..", 0),          # only punctuation
            ("éß", 2),              # Unicode letters counted
            ("中A3!", 3),           # CJK + Latin + digit, '!' ignored
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
            ("ABC", False),     # all upper
            ("abc", False),     # all lower
            ("aBc", True),      # mixed ASCII
            ("iOS", True),      # lower + upper
            ("H2O", False),     # digits ignored; no lowercase letters
            ("NaCl", True),     # camelcase style
            ("", False),        # empty
            ("123", False),     # digits only
            ("__-__", False),   # punctuation only
            ("a-A", True),      # symbols ignored; still both lower+upper
        ],
    )
    def test_ascii_and_symbols(self, tok, expected):
        assert _has_lower_and_upper(tok) is expected

    @pytest.mark.parametrize(
        "tok,expected",
        [
            ("Éclair", True),   # accented upper + lowercase
            ("ßA", True),       # German sharp s (lower) + uppercase
            ("İi", True),       # Turkish capital dotted I + lowercase i
            ("Δx", True),       # Greek uppercase + Latin lowercase
            ("中A", False),     # CJK isalpha=True but caseless; no lowercase present
            ("中aA", True),     # caseless + lower + upper → True
        ],
    )
    def test_unicode_cases(self, tok, expected):
        assert _has_lower_and_upper(tok) is expected

# Token pattern for tests:
# - Captures a "token" as letters followed by letters/digits or common separators.
# - Includes '.' so we can verify trailing-punct trimming (e.g., "RAM.")
PAT = re.compile(r"(?P<tok>[A-Za-z][A-Za-z0-9&\-/\.]*)")


def collect(text: str, cfg: DetectorConfig, pat: re.Pattern[str]):
    return list(iter_candidates_with(text, cfg, pat))


class TestIterCandidatesWith:
    def test_all_caps_simple(self):
        cfg = DetectorConfig()
        text = "We ran it on the GPU and CPU yesterday."
        out = collect(text, cfg, PAT)
        surfaces = [s for s, _, _ in out]
        assert "GPU" in surfaces
        assert "CPU" in surfaces

    def test_mixed_case_relaxation_enabled(self):
        # "iOS": 2/3 letters uppercase ≈ 0.667. With mixed-case relaxation (0.5) it should pass.
        cfg = DetectorConfig(enable_mixed_case=True, require_caps_ratio_mixed=0.5)
        text = "We ship an iOS build every week."
        out = collect(text, cfg, PAT)
        surfaces = [s for s, _, _ in out]
        assert "iOS" in surfaces  # relaxed threshold applied (upp >= 2)

    def test_mixed_case_relaxation_disabled(self):
        # With relaxation OFF, require_caps_ratio=0.7 and iOS has ~0.667 → should be filtered out.
        cfg = DetectorConfig(enable_mixed_case=False)
        text = "We ship an iOS build every week."
        out = collect(text, cfg, PAT)
        surfaces = [s for s, _, _ in out]
        assert "iOS" not in surfaces

    def test_digits_ignored_in_caps_ratio(self):
        # "H2O": letters H,O are uppercase; digit '2' ignored → ratio = 1.0 → passes.
        cfg = DetectorConfig()
        text = "Check the H2O level."
        out = collect(text, cfg, PAT)
        surfaces = [s for s, _, _ in out]
        assert "H2O" in surfaces

    def test_trailing_punct_stripped_and_indices(self):
        cfg = DetectorConfig()
        text = "Memory uses RAM."
        out = collect(text, cfg, PAT)
        # Expect one candidate "RAM" with indices pointing exactly to 'RAM' (not the '.')
        assert any(s == "RAM" for s, _, _ in out)
        # Verify indices are tight to the token (no trailing '.')
        for srf, s, e in out:
            if srf == "RAM":
                assert text[s:e] == "RAM"
                # e should be the position right after 'M'
                assert text[e : e + 1] == "."  # the '.' is outside the candidate

    def test_min_len_enforced(self):
        # Default min_len=2 → single-letter tokens should be filtered.
        cfg = DetectorConfig()
        text = "A B CD"
        out = collect(text, cfg, PAT)
        surfaces = [s for s, _, _ in out]
        assert "A" not in surfaces
        assert "B" not in surfaces
        assert "CD" in surfaces

    def test_max_len_enforced(self):
        # Default max_len=10 → very long all-caps should be filtered.
        cfg = DetectorConfig()
        long_tok = "THISISVERYLONG"  # length 14
        text = f"Edge {long_tok} token."
        out = collect(text, cfg, PAT)
        surfaces = [s for s, _, _ in out]
        assert long_tok not in surfaces

    def test_allowed_separators_compound_tokens(self):
        # Ensure tokens with separators (&, -) get considered and pass.
        cfg = DetectorConfig()
        text = "Our R&D team ported GPU-CPU pipelines."
        out = collect(text, cfg, PAT)
        surfaces = [s for s, _, _ in out]
        assert "R&D" in surfaces
        assert "GPU-CPU" in surfaces

    def test_mixed_case_requires_two_uppers_for_relax(self):
        # Relaxation only kicks in if upp >= 2. "eBay" has only 1 uppercase in practice (B),
        # so it should fail under default require_caps_ratio=0.7.
        cfg = DetectorConfig(enable_mixed_case=True)
        text = "We listed it on eBay."
        out = collect(text, cfg, PAT)
        surfaces = [s for s, _, _ in out]
        assert "eBay" not in surfaces

    def test_mixed_case_relax_threshold_param(self):
        # Tighten the mixed-case threshold so "NaCl" (2/4 = 0.5) fails.
        cfg = DetectorConfig(enable_mixed_case=True, require_caps_ratio_mixed=0.6)
        text = "We used NaCl in the experiment."
        out = collect(text, cfg, PAT)
        surfaces = [s for s, _, _ in out]
        assert "NaCl" not in surfaces
