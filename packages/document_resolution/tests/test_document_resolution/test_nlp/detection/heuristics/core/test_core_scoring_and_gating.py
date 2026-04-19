from types import SimpleNamespace

import pytest

import document_resolution.nlp.detection.acronym.detector as det
import document_resolution.nlp.detection.heuristics.core as core
from document_resolution.nlp.common.types import AcronymDetectorConfig, Span
from document_resolution.nlp.detection.heuristics.core import (
    _has_lower_and_upper,
    context_window,
)


def _idx(text: str, token: str) -> Span:
    s = text.index(token)
    return s, s + len(token)


class TestAcceptCandidate:
    def test_trailing_punct_is_stripped_and_returns_span(self, monkeypatch):
        text = "ABC!"
        cfg = DummyCfg(min_len=3, max_len=10, require_caps_ratio=0.8)

        # strip trailing '!' -> (0,3)
        monkeypatch.setattr(core, "strip_trailing_punct_span", lambda t, s, e: (s, e - 1), raising=False)
        monkeypatch.setattr(core, "has_letter", lambda s: True, raising=False)
        monkeypatch.setattr(core, "core_len_for_bounds", lambda s: len(s), raising=False)
        monkeypatch.setattr(core, "caps_ratio", lambda s: 1.0, raising=False)
        monkeypatch.setattr(core, "_has_lower_and_upper", lambda s: False, raising=False)

        out = core._accept_candidate(text, cfg, 0, len(text))
        assert out == ("ABC", 0, 3)

    def test_rejects_when_no_letters(self, monkeypatch):
        text = "123-456"
        cfg = DummyCfg(min_len=2)

        monkeypatch.setattr(core, "strip_trailing_punct_span", lambda t, s, e: (s, e), raising=False)
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

        monkeypatch.setattr(core, "strip_trailing_punct_span", lambda t, _s, _e: (s, e), raising=False)
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

        monkeypatch.setattr(core, "strip_trailing_punct_span", lambda t, _s, _e: (s, e), raising=False)
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

        monkeypatch.setattr(core, "strip_trailing_punct_span", lambda t, _s, _e: (s, e), raising=False)
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
        monkeypatch.setattr(core, "strip_trailing_punct_span", lambda t, _s, _e: (s, e), raising=False)
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

        # strip_trailing_punct_span is still called (function calls it before the raw len check),
        # but the early length check will return None before any deeper gates.
        monkeypatch.setattr(core, "strip_trailing_punct_span", strip_fn, raising=False)
        monkeypatch.setattr(core, "has_letter", lambda s: True, raising=False)
        monkeypatch.setattr(core, "core_len_for_bounds", lambda s: 5, raising=False)
        monkeypatch.setattr(core, "caps_ratio", lambda s: 1.0, raising=False)
        monkeypatch.setattr(core, "_has_lower_and_upper", lambda s: False, raising=False)

        out = core._accept_candidate(text, cfg, 0, 5)
        assert out is None
        assert calls["strip_called"] is True

    def _span(self, text: str, token: str) -> Span:
        s = text.index(token)
        return s, s + len(token)

    def test_end_to_end_accept_reject_matrix(self):
        # Configure with realistic bounds and mixed-case relaxation.
        cfg = AcronymDetectorConfig(
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
        assert text[left:right].startswith(text[start - 3 : start])

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


class DummyCfg(AcronymDetectorConfig):
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


class TestScoreUnit:
    @pytest.fixture
    def patch_score_cues(self, monkeypatch):
        def _apply(*, in_brackets=(False, False), paren_def=False, stands_for=False):
            monkeypatch.setattr(core, "in_brackets", lambda t, s, e: in_brackets, raising=True)
            monkeypatch.setattr(core, "has_paren_definition", lambda t, e: paren_def, raising=True)
            monkeypatch.setattr(core, "has_stands_for_follow", lambda t, e, max_chars=24: stands_for, raising=True)

        return _apply

    def test_base_score_no_signals(self, patch_score_cues):
        patch_score_cues()
        cfg = AcronymDetectorConfig()
        text = "We use GPU daily."
        s, e = _idx(text, "GPU")
        assert core.calc_score("GPU", text, s, e, cfg) == 0.6

    def test_in_brackets_inside_adds_point_25(self, patch_score_cues):
        patch_score_cues(in_brackets=(True, False))
        cfg = AcronymDetectorConfig()
        text = "(GPU) is fast."
        s, e = _idx(text, "GPU")
        assert core.calc_score("GPU", text, s, e, cfg) == 0.6 + 0.25

    def test_inside_takes_precedence_over_adjacent(self, patch_score_cues):
        patch_score_cues(in_brackets=(True, False))
        cfg = AcronymDetectorConfig()
        text = "GPU near brackets."
        s, e = _idx(text, "GPU")
        assert core.calc_score("GPU", text, s, e, cfg) == 0.85

    def test_paren_definition_adds_point_25(self, patch_score_cues):
        patch_score_cues(in_brackets=(True, True))
        cfg = AcronymDetectorConfig()
        text = "GPU (Graphics Processing Unit)"
        s, e = _idx(text, "GPU")
        assert core.calc_score("GPU", text, s, e, cfg) == 0.6 + 0.25

    def test_stands_for_follow_adds_point_15(self, patch_score_cues):
        patch_score_cues(in_brackets=(False, False), paren_def=False, stands_for=True)
        cfg = AcronymDetectorConfig()
        text = "GPU stands for Graphics Processing Unit."
        s, e = _idx(text, "GPU")
        assert core.calc_score("GPU", text, s, e, cfg) == 0.6 + 0.15

    def test_soft_blacklist_penalises_point_2(self, patch_score_cues):
        patch_score_cues()
        cfg = AcronymDetectorConfig()
        text = "We saw AS today."
        s, e = _idx(text, "AS")
        # AS is in cfg.soft_blacklist → -0.2
        assert core.calc_score("AS", text, s, e, cfg) == 0.6 - 0.2

    def test_upper_bound_clamped_to_one(self, patch_score_cues):
        patch_score_cues(in_brackets=(True, False), paren_def=True, stands_for=True)

        cfg = AcronymDetectorConfig()
        text = "GPU (Graphics Processing Unit). GPU stands for Graphics Processing Unit."
        s, e = _idx(text, "GPU")
        assert core.calc_score("GPU", text, s, e, cfg) == 1.0


class TestScoreIntegration:
    def test_score_with_real_heuristics_stands_for(self):
        # No patching: rely on real implementations.
        # Pattern: "<TOKEN> stands for <definition>" should add +0.15.
        text = "In docs, GPU stands for Graphics Processing Unit."
        s, e = _idx(text, "GPU")
        val = det.calc_score("GPU", text, s, e, AcronymDetectorConfig())

        # Expect base 0.6 + 0.15 for 'stands for', possibly more if in_brackets
        # logic treats proximity to punctuation as adjacent—but there are no brackets here.
        assert val == 0.75


class TestBoostConfidenceIfWhitelisted:
    def _cfg(self, **overrides):
        # Minimal, flexible config object
        base = {
            "whitelist_two_letter": {"AI", "UK"},
            "two_letter_boost": 0.75,
            "dotted_display": "strip",
            "allow_chars": "&-/.",
        }
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_boosts_when_two_letter_and_whitelisted(self, monkeypatch):
        # Force the normalized key to 'AI'
        monkeypatch.setattr(core, "normalize_acronym_key", lambda surface, **_: "AI", raising=True)

        cfg = self._cfg()
        result = core.boost_confidence_if_whitelisted("A.I.", 0.20, cfg)
        assert result == pytest.approx(0.95)  # 0.20 + 0.75

    def test_caps_at_point_99(self, monkeypatch):
        monkeypatch.setattr(core, "normalize_acronym_key", lambda surface, **_: "AI", raising=True)

        cfg = self._cfg(two_letter_boost=0.75)
        result = core.boost_confidence_if_whitelisted("AI", 0.50, cfg)
        assert result == pytest.approx(0.99)  # capped

    def test_no_boost_when_not_whitelisted(self, monkeypatch):
        monkeypatch.setattr(
            core,
            "normalize_acronym_key",
            lambda surface, **_: "TV",  # not in whitelist
            raising=True,
        )

        cfg = self._cfg()
        result = core.boost_confidence_if_whitelisted("TV", 0.40, cfg)
        assert result == pytest.approx(0.40)

    def test_no_boost_when_not_two_letters(self, monkeypatch):
        # Even if present in whitelist, length != 2 should not boost
        monkeypatch.setattr(core, "normalize_acronym_key", lambda surface, **_: "GPU", raising=True)

        cfg = self._cfg(whitelist_two_letter={"GPU"})  # irrelevant; len != 2
        result = core.boost_confidence_if_whitelisted("GPU", 0.33, cfg)
        assert result == pytest.approx(0.33)

    def test_respects_custom_boost_from_cfg(self, monkeypatch):
        monkeypatch.setattr(core, "normalize_acronym_key", lambda surface, **_: "UK", raising=True)

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
        cfg = SimpleNamespace(whitelist_two_letter={"AI"}, allow_chars="&-/.", dotted_display="strip")
        result = core.boost_confidence_if_whitelisted("A.I.", 0.10, cfg)

        # Default boost = 0.75 ⇒ 0.85
        assert result == pytest.approx(0.85)
        # Function should fall back to defaults inside getattr calls
        assert seen.get("allow_chars") == "&-/."
        assert seen.get("dotted_mode") == "strip"
