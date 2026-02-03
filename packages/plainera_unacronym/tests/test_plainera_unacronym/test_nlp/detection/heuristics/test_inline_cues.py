# tests/test_plainera_unacronym/test_nlp/detection/test_detector/test_inline_cues.py
import re

import pytest

from plainera_unacronym.nlp.detection.heuristics.inline_cues import (
    _compile_inline_cues_pattern,
    boost_confidence_if_inline_cue,
)


class TestCompileInlineCuesPattern:
    def test_compiles_case_insensitive_and_matches_whole_phrase(self):
        pat = _compile_inline_cues_pattern((r"stands?\s+for",))
        assert pat.flags & re.IGNORECASE, f"IGNORECASE not set; flags={pat.flags}"
        expected = re.IGNORECASE | re.UNICODE
        assert (pat.flags & expected) == expected

        # whole phrase match (case-insensitive)
        assert pat.search("StAnDs for ") is not None
        assert pat.search("stands for,") is not None

    def test_requires_right_side_separator_via_lookahead(self):
        pat = _compile_inline_cues_pattern((r"means",))

        assert pat.search("means ") is not None
        assert pat.search("means,") is not None
        assert pat.search("means-") is not None
        assert pat.search("means—") is not None

        # Should NOT match when immediately followed by a letter (fails lookahead)
        assert pat.search("meansX") is None


class TestBoostConfidenceIfInlineCue:
    def test_does_not_boost_for_long_surfaces(self):
        text = "NATO stands for North Atlantic Treaty Organization."
        e = text.index("NATO") + len("NATO")
        assert boost_confidence_if_inline_cue("NATO", text, e, 0.50) == 0.50

    @pytest.mark.parametrize(
        "text,surface",
        [
            ("AM, short for ante meridiem.", "AM"),
            ("NLP stands for Natural language processing.", "NLP"),
            ("GPU is an acronym for graphics processing unit.", "GPU"),
            ("API Abbreviated as application programming interface.", "API"),
            ("RAM MEANS random access memory.", "RAM"),
        ],
    )
    def test_boosts_when_cue_is_to_the_right_case_insensitive(self, text, surface):
        e = text.index(surface) + len(surface)
        out = boost_confidence_if_inline_cue(surface, text, e, 0.50)
        assert out == pytest.approx(0.70)

    def test_caps_at_point_99(self):
        text = "NLP stands for Natural language processing."
        e = text.index("NLP") + 3
        out = boost_confidence_if_inline_cue("NLP", text, e, 0.90)
        assert out == pytest.approx(0.99)

    def test_does_not_boost_if_cue_is_left_of_end_offset(self):
        text = "stands for NLP is sometimes used."
        e = text.index("NLP") + 3
        # cue occurs before e; right slice should not include it
        assert boost_confidence_if_inline_cue("NLP", text, e, 0.50) == 0.50

    def test_does_not_boost_if_cue_is_beyond_right_window(self):
        surface = "NLP"
        text = "NLP" + (" " * 80) + "stands for Natural language processing."
        e = text.index(surface) + len(surface)

        # cue is > 60 chars away, so it must not be seen
        assert boost_confidence_if_inline_cue(surface, text, e, 0.50) == 0.50
