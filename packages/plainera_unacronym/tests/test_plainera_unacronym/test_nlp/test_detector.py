import pytest
from dataclasses import dataclass

from plainera_unacronym.nlp import Occurrence, DetectorConfig
from plainera_unacronym.nlp.detector import _build_occurrence_from_match
import plainera_unacronym.nlp.detector as det


@dataclass(frozen=True, slots=True)
class _Cfg(DetectorConfig):
    allow_chars: str = "-./"
    window_chars: int = 80
    dotted_display: str = "strip"         # "strip" | "preserve"
    debug_reasons: bool = False


class TestBuildOccurrenceFromMatch:
    def test_strip_mode_does_not_include_trailing_dot(self, monkeypatch):
        """
        When dotted_display='strip' and the next char is '.', we should NOT extend the end,
        and the displayed surface/acronym should NOT include the dot.
        """
        calls = {}

        def fake_normalize_key(base, allow_chars, dotted_mode):
            calls["normalize_key"] = (base, allow_chars, dotted_mode)
            return f"NK[{base}|{dotted_mode}]"

        def fake_context_window(text, s, e, win):
            calls["context_window"] = (text, s, e, win)
            return (111, 222)

        # No need to mock strip_terminal_plural: test uses a form without plural suffix.
        monkeypatch.setattr(
            "plainera_unacronym.nlp.detector.normalize_key", fake_normalize_key, raising=True
        )
        monkeypatch.setattr(
            "plainera_unacronym.nlp.detector.context_window", fake_context_window, raising=True
        )

        cfg = _Cfg(dotted_display="strip")
        text = "NASA."
        surface = "NASA"
        s, e = 0, 4  # 'NASA' ends at 4, '.' is at index 4

        occ, display_key = _build_occurrence_from_match(cfg, text, surface, s, e, conf=0.87)

        # display key returned from our fake normalizer
        assert display_key == "NK[NASA|strip]"

        # Occurrence fields
        assert isinstance(occ, Occurrence)
        assert occ.acronym == "NASA"           # dot NOT included
        assert occ.start_offset == 0
        assert occ.end_offset == 4             # NOT advanced
        assert occ.confidence == 0.87
        assert occ.context_window == (111, 222)
        assert occ.normalized_key == display_key
        assert occ.reasons is None

        # Assert helper call args
        assert calls["normalize_key"] == ("NASA", cfg.allow_chars, "strip")
        assert calls["context_window"] == (text, s, 4, cfg.window_chars)

    def test_preserve_mode_includes_trailing_dot(self, monkeypatch):
        """
        When dotted_display='preserve' and the next char is '.',
        we must include it in the displayed surface and advance the end offset by 1.
        """
        calls = {}

        def fake_normalize_key(base, allow_chars, dotted_mode):
            calls["normalize_key"] = (base, allow_chars, dotted_mode)
            return f"NK[{base}|{dotted_mode}]"

        def fake_context_window(text, s, e, win):
            calls["context_window"] = (text, s, e, win)
            return (5, 10)

        monkeypatch.setattr(
            "plainera_unacronym.nlp.detector.normalize_key", fake_normalize_key, raising=True
        )
        monkeypatch.setattr(
            "plainera_unacronym.nlp.detector.context_window", fake_context_window, raising=True
        )

        cfg = _Cfg(dotted_display="preserve")
        text = "NASA."
        surface = "NASA"
        s, e = 0, 4  # '.' at index 4

        occ, display_key = _build_occurrence_from_match(cfg, text, surface, s, e, conf=0.9)

        assert display_key == "NK[NASA.|preserve]"
        assert occ.acronym == "NASA."          # dot INCLUDED
        assert occ.end_offset == 5             # advanced
        assert occ.context_window == (5, 10)

        # Helper call args used the adjusted end offset
        assert calls["normalize_key"] == ("NASA.", cfg.allow_chars, "preserve")
        assert calls["context_window"] == (text, s, 5, cfg.window_chars)

    def test_debug_reasons_attached_when_enabled(self, monkeypatch):
        """
        When cfg.debug_reasons=True, reasons should be attached as a tuple from reason_tags(...).
        Also verify preserve mode advances end to include the trailing '.'.
        """
        import plainera_unacronym.nlp.detector as det

        def fake_normalize_key(base, allow_chars, dotted_mode):
            return "NK"

        def fake_context_window(text, s, e, win):
            return (0, 7)

        def fake_reason_tags(surface, text, s, e_passed, cfg):
            # Assert we got the adjusted end (includes the trailing dot)
            assert e_passed == e + 1
            return ["EDGE", "PUNCT"]

        monkeypatch.setattr(det, "normalize_key", fake_normalize_key, raising=True)
        monkeypatch.setattr(det, "context_window", fake_context_window, raising=True)
        monkeypatch.setattr(det, "reason_tags", fake_reason_tags, raising=True)

        cfg = _Cfg(dotted_display="preserve", debug_reasons=True)
        text = "N.A.S.A."
        s = text.index("N.A.S.A")
        e = s + len("N.A.S.A")  # == 7; text[e] is the '.' after the matched surface
        surface = text[s:e]  # "N.A.S.A"

        occ, display_key = _build_occurrence_from_match(cfg, text, surface, s, e, conf=0.7)

        assert display_key == "NK"
        assert occ.reasons == ("EDGE", "PUNCT")  # tuple-ized
        assert occ.acronym == "N.A.S.A."  # dot included in display
        assert occ.end_offset == e + 1  # advanced to include final dot (== 8)

    def test_returns_occurrence_and_display_key_tuple(self, monkeypatch):
        # Patch where the function looks up the names (the detector module)
        monkeypatch.setattr(
            det,
            "normalize_key",
            lambda base, allow_chars, dotted_mode=None: f"{base.lower()}::{dotted_mode}",
            raising=True,
        )
        monkeypatch.setattr(
            det,
            "context_window",
            lambda text, s, e, w: (1, 2),
            raising=True,
        )

        cfg = _Cfg(dotted_display="strip")
        text = "API."
        surface = "API"
        s, e = 0, 3

        occ, key = _build_occurrence_from_match(cfg, text, surface, s, e, conf=0.5)

        assert key == "api::strip"
        assert isinstance(occ, Occurrence)
        assert occ.normalized_key == key

    # ----------- INTEGRATION TESTS BELOW -----------

    def test_strip_vs_preserve_and_plural_and_key(self):
        text = "We tested GPUs. Also NASA."
        s_gpu = text.index("GPUs")
        e_gpu = s_gpu + len("GPUs")  # '.' follows immediately in text

        # strip mode → do NOT include trailing dot; plural stripped before normalisation
        cfg_strip = DetectorConfig(dotted_display="strip")
        occ_s, key_s = _build_occurrence_from_match(
            cfg_strip, text, text[s_gpu:e_gpu], s_gpu, e_gpu, conf=0.91
        )

        assert isinstance(occ_s, Occurrence)
        assert occ_s.acronym == "GPU"  # plural removed
        assert occ_s.end_offset == e_gpu  # trailing '.' NOT included
        assert occ_s.normalized_key == key_s
        # normalize_key in strip mode removes dots; no dots present anyway, key equals base
        assert key_s == "GPU"

        # preserve mode → include trailing dot in surface & end_offset
        cfg_pres = DetectorConfig(dotted_display="preserve")
        occ_p, key_p = _build_occurrence_from_match(
            cfg_pres, text, text[s_gpu:e_gpu], s_gpu, e_gpu, conf=0.91
        )
        assert occ_p.acronym == "GPUs."  # dot included
        assert occ_p.end_offset == e_gpu + 1  # advanced
        assert occ_p.normalized_key == key_p
        # In preserve mode, key keeps the dot
        assert key_p.endswith(".")

        # Context window invariants (don’t assert exact indices across implementations)
        l_s, r_s = occ_s.context_window
        l_p, r_p = occ_p.context_window
        n = len(text)
        for l, r, st, en in [(l_s, r_s, occ_s.start_offset, occ_s.end_offset),
                             (l_p, r_p, occ_p.start_offset, occ_p.end_offset)]:
            assert 0 <= l < r <= n
            assert l <= st < en <= r


    def test_reason_tags_inside_parens_and_dotted(self):
        text = "The (N.A.S.A.) rocket launched."
        s = text.index("N.A.S.A")
        e = s + len("N.A.S.A")
        cfg = DetectorConfig(dotted_display="preserve", debug_reasons=True, enable_dotted=True)

        occ, _ = _build_occurrence_from_match(cfg, text, text[s:e], s, e, conf=0.7)

        assert isinstance(occ.reasons, tuple) and occ.reasons
        assert "inside_parens" in occ.reasons
        assert "dotted_initialism" in occ.reasons
        # Do NOT assert next_word_lowercase here, because the next char is ')'.

    def test_reason_tags_next_word_lowercase_when_followed_by_space_word(self):
        text = "N.A.S.A. rocket launched."
        s = text.index("N.A.S.A")
        e = s + len("N.A.S.A")
        cfg = DetectorConfig(dotted_display="preserve", debug_reasons=True, enable_dotted=True)

        occ, _ = _build_occurrence_from_match(cfg, text, text[s:e], s, e, conf=0.7)

        assert isinstance(occ.reasons, tuple) and occ.reasons
        assert "dotted_initialism" in occ.reasons
        # Here, after the preserved '.', the next char is a space then 'r' → lowercase tag expected.
        assert "next_word_lowercase" in occ.reasons

    def test_context_window_bounds_exact_in_simple_sentence(self):
        # One sentence; window should start at the sentence start and end at the terminator (included).
        text = "Hello world. API is here!"
        s = text.index("API")
        e = s + 3
        cfg = DetectorConfig(window_chars=80, dotted_display="strip")

        occ, _ = _build_occurrence_from_match(cfg, text, text[s:e], s, e, conf=0.5)

        # Sentence is " API is here!" after the period+space
        # Compute expected left: after the period space; right: include '!'
        left_expected = text.index("API")  # because the sentence delimiter is right before 'API '
        # We expect the window to include up to and including the '!'
        right_expected = len(text)  # '!' is last char; function includes the terminator
        assert occ.context_window == (left_expected, right_expected)
