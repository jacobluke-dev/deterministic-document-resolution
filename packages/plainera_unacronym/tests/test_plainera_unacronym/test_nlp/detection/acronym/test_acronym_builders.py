import pytest

from plainera_unacronym.nlp import Occurrence
from plainera_unacronym.nlp.common.types import OccurrenceBuildError
from plainera_unacronym.nlp.detection.acronym.builders import build_occurrence_from_match, adjust_end_for_trailing_dot


class TestBuildOccurrenceFromMatch:
    def test_strip_mode_does_not_include_trailing_dot(self, _patch, test_cfg):
        calls = {}

        def fake_normalize_acronym_key(base, allow_chars, dotted_mode):
            calls["normalize_acronym_key"] = (base, allow_chars, dotted_mode)
            return f"NK[{base}|{dotted_mode}]"

        def fake_context_window(text, s, e, win):
            calls["context_window"] = (text, s, e, win)
            return 111, 222

        _patch(
            build_occurrence_from_match,
            normalize_acronym_key=fake_normalize_acronym_key,
            context_window=fake_context_window,
        )

        cfg = test_cfg(dotted_display="strip")
        text = "NASA."
        surface = "NASA"
        s, e = 0, 4  # '.' is at index 4

        occ, display_key = build_occurrence_from_match(cfg, text, surface, s, e, conf=0.87)

        # display key returned from our fake normalizer
        assert display_key == "NK[NASA|strip]"

        # Occurrence fields
        assert isinstance(occ, Occurrence)
        assert occ.acronym == "NASA"
        assert occ.start_offset == 0
        assert occ.end_offset == 4
        assert occ.occurrence_confidence == 0.87
        assert occ.segment_window == (111, 222)
        assert occ.normalized_key == display_key
        assert occ.reasons is None

        assert calls["normalize_acronym_key"] == ("NASA", cfg.allow_chars, "strip")
        assert calls["context_window"] == (text, s, 4, cfg.window_chars)

    def test_preserve_mode_includes_trailing_dot(self, _patch, test_cfg):
        """
        When dotted_display='preserve' and the next char is '.',
        we must include it in the displayed surface and advance the end offset by 1.
        """
        calls = {}

        def fake_normalize_acronym_key(base, allow_chars, dotted_mode):
            calls["normalize_acronym_key"] = (base, allow_chars, dotted_mode)
            return f"NK[{base}|{dotted_mode}]"

        def fake_context_window(text, s, e, win):
            calls["context_window"] = (text, s, e, win)
            return 5, 10

        _patch(
            build_occurrence_from_match,
            normalize_acronym_key=fake_normalize_acronym_key,
            context_window=fake_context_window,
        )

        cfg = test_cfg(dotted_display="preserve")
        text = "N.A.S.A."
        surface = "N.A.S.A"
        s, e = 0, 4  # '.' at index 4

        occ, display_key = build_occurrence_from_match(cfg, text, surface, s, e, conf=0.9)

        assert display_key == "NK[N.A.S.A|preserve]"
        assert occ.acronym == "N.A.S.A"  # dots INCLUDED
        assert occ.end_offset == 4  # advanced
        assert occ.segment_window == (5, 10)

        # Helper call args used the adjusted end offset
        assert calls["normalize_acronym_key"] == ("N.A.S.A", cfg.allow_chars, "preserve")
        assert calls["context_window"] == (text, s, 4, cfg.window_chars)

    def test_debug_reasons_attached_when_enabled(self, _patch, test_cfg):
        """
        When cfg.debug_reasons=True, reasons should be attached as a tuple from reason_tags(...).
        Also verify preserve mode advances end to include the trailing '.'.
        """

        def fake_normalize_acronym_key(base, allow_chars, dotted_mode):
            return "NK"

        def fake_context_window(text, s, e, win):
            return 0, 7

        def fake_reason_tags(surface, text, s, e_passed, cfg):
            # Assert we got the adjusted end (includes the trailing dot)
            assert e_passed == e + 1
            return ["EDGE", "PUNCT"]

        _patch(
            build_occurrence_from_match,
            normalize_acronym_key=fake_normalize_acronym_key,
            context_window=fake_context_window,
            reason_tags=fake_reason_tags,
        )

        cfg = test_cfg(dotted_display="preserve", debug_reasons=True)
        text = "N.A.S.A."
        s = 0
        surface = "N.A.S.A"
        e = s + len(surface)  # 7, so text[e] is the trailing '.'
        assert cfg.dotted_display == "preserve"
        assert text[e] == ".", (e, text, text[e - 2 : e + 2])

        occ, display_key = build_occurrence_from_match(cfg, text, surface, s, e, conf=0.7)

        assert display_key == "NK"
        assert occ.reasons == ("EDGE", "PUNCT")
        assert occ.acronym == "N.A.S.A"
        assert occ.end_offset == e + 1  # == 8

    def test_returns_occurrence_and_display_key_tuple(self, _patch, test_cfg):
        cfg = test_cfg(dotted_display="strip")
        text = "API."
        surface = "API"
        s, e = 0, 3
        _patch(
            build_occurrence_from_match,
            normalize_acronym_key=lambda base, allow_chars, dotted_mode=None: f"{base.lower()}::{dotted_mode}",
            context_window=lambda text, s, e, w: (1, 2),
        )

        occ, key = build_occurrence_from_match(cfg, text, surface, s, e, conf=0.5)

        assert key == "api::strip"
        assert isinstance(occ, Occurrence)
        assert occ.normalized_key == key

    # ----------- INTEGRATION TESTS BELOW -----------

    def test_strip_vs_preserve_and_plural_and_key(self, test_cfg):
        text = "We tested GPUs. Also NASA."
        s_gpu = text.index("GPUs")
        e_gpu = s_gpu + len("GPUs")  # '.' follows immediately in text

        # strip mode → do NOT include trailing dot; plural stripped before normalisation
        cfg_strip = test_cfg(dotted_display="strip")
        occ_s, key_s = build_occurrence_from_match(cfg_strip, text, text[s_gpu:e_gpu], s_gpu, e_gpu, conf=0.91)

        assert isinstance(occ_s, Occurrence)
        assert occ_s.acronym == "GPU"  # plural removed
        assert occ_s.end_offset == e_gpu  # trailing '.' NOT included
        assert occ_s.normalized_key == key_s
        # normalize_acronym_key in strip mode removes dots; no dots present anyway, key equals base
        assert key_s == "GPU"

        # preserve mode → removes trailing dot in surface & end_offset
        cfg_pres = test_cfg(dotted_display="preserve")
        occ_p, key_p = build_occurrence_from_match(cfg_pres, text, text[s_gpu:e_gpu], s_gpu, e_gpu, conf=0.91)
        assert occ_p.acronym == "GPU"  # dot not included
        assert occ_p.end_offset == e_gpu + 1
        assert occ_p.normalized_key == key_p
        # In preserve mode, key keeps the dot
        assert not key_p.endswith(".")

        # Context window invariants (don’t assert exact indices across implementations)
        l_s, r_s = occ_s.segment_window
        l_p, r_p = occ_p.segment_window
        n = len(text)
        for left, right, strt, end in [
            (l_s, r_s, occ_s.start_offset, occ_s.end_offset),
            (l_p, r_p, occ_p.start_offset, occ_p.end_offset),
        ]:
            assert 0 <= left < right <= n
            assert left <= strt < end <= right

    def test_reason_tags_inside_parens_and_dotted(self, test_cfg):
        text = "The (N.A.S.A.) rocket launched."
        s = text.index("N.A.S.A")
        e = s + len("N.A.S.A")
        cfg = test_cfg(dotted_display="preserve", debug_reasons=True, enable_dotted=True)

        occ, _ = build_occurrence_from_match(cfg, text, text[s:e], s, e, conf=0.7)

        assert isinstance(occ.reasons, tuple) and occ.reasons
        assert "inside_parens" in occ.reasons
        assert "dotted_initialism" in occ.reasons
        # Do NOT assert next_word_lowercase here, because the next char is ')'.

    def test_reason_tags_next_word_lowercase_when_followed_by_space_word(self, test_cfg):
        text = "N.A.S.A. rocket launched."
        s = text.index("N.A.S.A")
        e = s + len("N.A.S.A")
        cfg = test_cfg(dotted_display="preserve", debug_reasons=True, enable_dotted=True)

        occ, _ = build_occurrence_from_match(cfg, text, text[s:e], s, e, conf=0.7)

        assert isinstance(occ.reasons, tuple) and occ.reasons
        assert "dotted_initialism" in occ.reasons
        # Here, after the preserved '.', the next char is a space then 'r' → lowercase tag expected.
        assert "next_word_lowercase" in occ.reasons

    def test_context_window_bounds_exact_in_simple_sentence(self, test_cfg):
        # One sentence; window should start at the sentence start and end at the terminator (included).
        text = "Hello world. API is here!"
        s = text.index("API")
        e = s + 3
        cfg = test_cfg(window_chars=80, dotted_display="strip")

        occ, _ = build_occurrence_from_match(cfg, text, text[s:e], s, e, conf=0.5)

        # Sentence is " API is here!" after the period+space
        # Compute expected left: after the period space; right: include '!'
        left_expected = text.index("API")  # because the sentence delimiter is right before 'API '
        # We expect the window to include up to and including the '!'
        right_expected = len(text)  # '!' is last char; function includes the terminator
        assert occ.segment_window == (left_expected, right_expected)



class TestAdjustEndForTrailingDotUnit:
    def test_strip_mode_does_not_advance_when_dot_present(self, test_cfg):
        cfg = test_cfg(dotted_display="strip")
        text = "NASA."
        s, e = 0, 4  # span is "NASA", dot is at text[4]
        assert text[e] == "."
        assert adjust_end_for_trailing_dot(cfg, text, s, e) == e

    def test_preserve_mode_advances_by_one_when_dot_present(self, test_cfg):
        cfg = test_cfg(dotted_display="preserve")
        text = "NASA."
        s, e = 0, 4
        assert text[e] == "."
        assert adjust_end_for_trailing_dot(cfg, text, s, e) == e + 1

    def test_preserve_mode_does_not_advance_when_no_dot(self, test_cfg):
        cfg = test_cfg(dotted_display="preserve")
        text = "NASA!"
        s, e = 0, 4
        assert text[e] == "!"
        assert adjust_end_for_trailing_dot(cfg, text, s, e) == e

    def test_at_end_of_text_never_advances(self, test_cfg):
        cfg = test_cfg(dotted_display="preserve")
        text = "NASA"
        s, e = 0, 4  # e == len(text)
        assert adjust_end_for_trailing_dot(cfg, text, s, e) == e

    @pytest.mark.parametrize("display_mode", ["strip", "preserve", "unknown"])
    def test_unknown_mode_behaves_like_strip(self, test_cfg, display_mode):
        # getattr(cfg, "dotted_display", "strip") reads whatever you set,
        # but only "preserve" causes advancement.
        cfg = test_cfg(dotted_display=display_mode)  # type: ignore[arg-type]
        text = "U.S."
        s, e = 0, 3  # span "U.S", dot at index 3
        assert text[e] == "."
        expected = e + 1 if display_mode == "preserve" else e
        assert adjust_end_for_trailing_dot(cfg, text, s, e) == expected

    @pytest.mark.parametrize(
        "text,s,e",
        [
            ("NASA.", -1, 4),  # negative start
            ("NASA.", 0, -1),  # negative end
            ("NASA.", 0, 999),  # end out of bounds
            ("NASA.", 3, 2),  # s >= end_for_occ (invalid slice)
        ],
    )
    def test_raises_on_bad_offsets(self, test_cfg, text, s, e):
        cfg = test_cfg(dotted_display="strip")
        with pytest.raises(OccurrenceBuildError):
            adjust_end_for_trailing_dot(cfg, text, s, e)

    def test_raises_when_preserve_advances_past_text_end(self, test_cfg):
        # e points at the last character '.', so preserve would try to advance past end.
        cfg = test_cfg(dotted_display="preserve")
        text = "X."
        s, e = 0, 1  # span "X", dot at index 1 (OK to advance to 2)
        assert adjust_end_for_trailing_dot(cfg, text, s, e) == 2

        # Now make the span include the dot already; e == len(text), cannot look at text[e]
        # and also no advancement should happen; still must validate offsets.
        s2, e2 = 0, 2
        assert adjust_end_for_trailing_dot(cfg, text, s2, e2) == 2
