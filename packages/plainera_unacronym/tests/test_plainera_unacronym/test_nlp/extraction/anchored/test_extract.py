import re
from types import SimpleNamespace

import plainera_unacronym.nlp.extraction.anchored.extract as mod
import pytest
from plainera_unacronym.nlp.common.types import ExtractedDefinition, FirstOccurrence
from plainera_unacronym.nlp.extraction import ExtractionConfig


def _fo(acr: str, start: int, end: int, *, norm: str | None = None):
    # Use your existing helper if you already have one
    from plainera_unacronym.nlp.common.types import FirstOccurrence
    return FirstOccurrence(acronym=acr, start_offset=start, end_offset=end, confidence=0.9, normalized_key=norm)


class TestBuildLocalWindowUnit:
    def test_middle_of_text_respects_windows(self):
        text = "0123456789ABCDEFGHIJ"
        fo = _fo("X", start=10, end=11)  # points at "A"
        left, right, seg = mod._build_local_window(text, fo, window_left=3, window_right=4)
        assert left == 7
        assert right == 15
        assert seg == text[7:15]

    def test_left_clamps_to_zero(self):
        text = "abcdef"
        fo = _fo("X", start=1, end=2)
        left, right, seg = mod._build_local_window(text, fo, window_left=50, window_right=0)
        assert left == 0
        assert right == 2
        assert seg == "ab"

    def test_right_clamps_to_len(self):
        text = "abcdef"
        fo = _fo("X", start=4, end=5)
        left, right, seg = mod._build_local_window(text, fo, window_left=0, window_right=50)
        assert left == 4
        assert right == len(text)
        assert seg == "ef"

    def test_both_sides_clamp(self):
        text = "abcdef"
        fo = _fo("X", start=0, end=6)
        left, right, seg = mod._build_local_window(text, fo, window_left=10, window_right=10)
        assert (left, right) == (0, 6)
        assert seg == text

    def test_window_can_be_zero_length(self):
        text = "abcdef"
        fo = _fo("X", start=2, end=4)  # "cd"
        left, right, seg = mod._build_local_window(text, fo, window_left=0, window_right=0)
        assert (left, right) == (2, 4)
        assert seg == "cd"

    def test_returns_segment_consistent_with_indices(self):
        text = "hello world"
        fo = _fo("X", start=6, end=11)  # "world"
        left, right, seg = mod._build_local_window(text, fo, window_left=2, window_right=2)
        assert seg == text[left:right]


class TestFoOccurrencePositionUnit:
    def test_returns_relative_offsets(self):
        fo = _fo("SSO", start=10, end=13)
        assert mod._fo_occurrence_position(fo, left=7) == (3, 6)

    def test_zero_left_is_identity(self):
        fo = _fo("PDF", start=2, end=5)
        assert mod._fo_occurrence_position(fo, left=0) == (2, 5)

    def test_left_equal_start_gives_zero_start(self):
        fo = _fo("GPU", start=20, end=23)
        assert mod._fo_occurrence_position(fo, left=20) == (0, 3)


def _ed(*, conf: float, d0: int, d1: int) -> ExtractedDefinition:
    return ExtractedDefinition(
        acronym="X",
        definition="DEF",
        source="in_text",
        confidence=conf,
        acr_start=0,
        acr_end=1,
        def_start=d0,
        def_end=d1,
        original_definition="DEF",
        kind="unknown",
    )


class TestPickBetterUnit:
    def test_none_best_returns_candidate(self):
        cand = _ed(conf=0.7, d0=0, d1=10)
        assert mod._pick_better(None, cand) is cand

    def test_higher_confidence_wins(self):
        best = _ed(conf=0.7, d0=0, d1=10)
        cand = _ed(conf=0.8, d0=0, d1=100)
        assert mod._pick_better(best, cand) is cand

    def test_tie_confidence_shorter_span_wins(self):
        best = _ed(conf=0.8, d0=0, d1=50)   # len 50
        cand = _ed(conf=0.8, d0=0, d1=10)   # len 10
        assert mod._pick_better(best, cand) is cand

    def test_tie_confidence_and_length_keeps_best(self):
        best = _ed(conf=0.8, d0=0, d1=10)
        cand = _ed(conf=0.8, d0=20, d1=30)  # same length
        assert mod._pick_better(best, cand) is best


class TestAnchoredConfidenceUnit:
    def test_no_distance_penalty(self):
        assert mod._anchored_confidence(base_conf=0.95, dist=0) == pytest.approx(0.95)

    def test_penalty_applies_linearly(self):
        # penalty = 100 * 0.0005 = 0.05
        assert mod._anchored_confidence(base_conf=0.95, dist=100) == pytest.approx(0.90)

    def test_distance_penalty_caps_at_200(self):
        # penalty capped at 200 => 0.1
        assert mod._anchored_confidence(base_conf=0.95, dist=9999) == pytest.approx(0.85)

    def test_confidence_caps_at_099(self):
        assert mod._anchored_confidence(base_conf=1.5, dist=0) == pytest.approx(0.99)

class TestDistanceFromFoUnit:
    def test_zero_when_aligned(self):
        assert mod._distance_from_fo(a0_local=5, left=10, fo_start_offset=15) == 0

    def test_positive_distance(self):
        # abs((5+10) - 20) = 5
        assert mod._distance_from_fo(a0_local=5, left=10, fo_start_offset=20) == 5

    def test_symmetric(self):
        # abs((5+10) - 12) = 3
        assert mod._distance_from_fo(a0_local=5, left=10, fo_start_offset=12) == 3


def _fo_extract_near_firsts_only(acr: str,
                                 text: str,
                                 *,
                                 norm: str | None = None,
                                 end_extra: int = 0) -> FirstOccurrence:
    a0 = text.index(acr)
    return FirstOccurrence(
        acronym=acr,
        start_offset=a0,
        end_offset=a0 + len(acr) + end_extra,
        confidence=0.9,
        normalized_key=norm,
    )


class TestExtractNearFirstsIntegration:
    def test_extracts_parenthetical_forward_form(self):
        text = "Single sign-on (SSO) is enabled."
        firsts = {"SSO": _fo_extract_near_firsts_only("SSO", text, norm="SSO")}
        cfg = ExtractionConfig()

        picks = mod.extract_near_firsts(text, firsts, window_left=80, window_right=80, cfg=cfg)

        assert picks["SSO"] is not None
        assert picks["SSO"].definition == "Single sign-on"
        assert text[picks["SSO"].acr_span[0]:picks["SSO"].acr_span[1]] == "SSO"



class TestExtractNearFirstsUnit:
    def test_returns_empty_for_no_firsts(self):
        out = mod.extract_near_firsts("anything", {}, window_left=10, window_right=10, cfg=ExtractionConfig())
        assert out == {}

    def test_picks_best_by_confidence_then_shorter_span(self, _patch):
        text = "Alpha (AAA) ... Beta (AAA)"
        fo = _fo_extract_near_firsts_only("AAA", text)
        firsts = {"AAA": fo}

        # Two “matches” against the same FO; select higher conf, or if tie, shorter def span.
        pat = re.compile(r"(?P<acr>AAA)")
        specs = [
            SimpleNamespace(pat=pat, base_conf=0.80, kind="inline", strategy="x"),
            SimpleNamespace(pat=pat, base_conf=0.80, kind="inline", strategy="x"),
        ]

        calls = {"n": 0}

        def fake_resolve_def_span(strategy, *, seg, m, acr_key, a1_local, cfg):
            # First candidate: longer; second candidate: shorter.
            calls["n"] += 1
            if calls["n"] == 1:
                return (0, 20)  # "Alpha (AAA) ... Bet"
            return (0, 5)      # "Alpha"

        _patch(
            mod.extract_near_firsts,
            compile_anchored_exact=lambda acr_surface, cfg: specs,
            resolve_def_span=fake_resolve_def_span,
            clean_definition=lambda orig, *, acr_norm, cfg, kind: orig,  # passthrough
        )

        out = mod.extract_near_firsts(text, firsts, window_left=999, window_right=999, cfg=ExtractionConfig())
        assert out["AAA"] is not None
        # Tie base_conf -> tie confidence -> shorter def span should win
        assert out["AAA"].definition == text[:5]  # "Alpha"

    def test_skips_non_aligned_matches(self, _patch):
        text = "X AAA Y"
        fo = _fo_extract_near_firsts_only("AAA", text)
        firsts = {"AAA": fo}

        # Pattern matches "AAA" but we’ll force mismatch by moving FO window offsets via fake build window.
        pat = re.compile(r"(?P<acr>AAA)")
        specs = [SimpleNamespace(pat=pat, base_conf=0.9, kind="inline", strategy="x")]

        _patch(
            mod.extract_near_firsts,
            compile_anchored_exact=lambda *_: specs,
            _build_local_window=lambda text, fo, wl, wr: (0, len(text), "ZZ AAA Y"),  # shifts match position
            resolve_def_span=lambda *a, **k: (0, 1),
            clean_definition=lambda *a, **k: "DEF",
        )

        out = mod.extract_near_firsts(text, firsts, window_left=10, window_right=10, cfg=ExtractionConfig())
        assert out["AAA"] is None

    def test_allows_trailing_dot_mismatch(self, _patch):
        # FO includes trailing dot; regex captures without it; should still accept.
        text = "U.S. Senate is a body."
        # FO spans "U.S." including the trailing dot (end_extra=1 because acr string below is "U.S")
        fo = _fo_extract_near_firsts_only("U.S", text, end_extra=1)
        firsts = {"U.S.": FirstOccurrence(
            acronym="U.S.",
            start_offset=fo.start_offset,
            end_offset=fo.end_offset,
            confidence=0.9,
            normalized_key="U.S."
        )}

        pat = re.compile(r"(?P<acr>U\.S)\.")  # captures without the trailing dot
        specs = [SimpleNamespace(pat=pat, base_conf=0.9, kind="inline", strategy="x")]

        _patch(
            mod.extract_near_firsts,
            compile_anchored_exact=lambda *_: specs,
            resolve_def_span=lambda *a, **k: (0, 3),
            clean_definition=lambda *a, **k: "US DEF",
        )

        out = mod.extract_near_firsts(text, firsts, window_left=50, window_right=50, cfg=ExtractionConfig())
        assert out["U.S."] is not None
        assert out["U.S."].definition == "US DEF"
        assert text[out["U.S."].acr_span[0]:out["U.S."].acr_span[1]] == "U.S."
