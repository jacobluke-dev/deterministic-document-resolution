from dataclasses import dataclass

import plainera_unacronym.nlp.detection.detector as det
import pytest
from plainera_unacronym.nlp import DetectorConfig, Occurrence
from plainera_unacronym.nlp.detection.detector import _build_occurrence_from_match, _score_chunk_worker


@dataclass(frozen=True, slots=True)
class _TestCfg(DetectorConfig):
    allow_chars: str = "&/-"
    window_chars: int = 80
    dotted_display: str = "strip"
    debug_reasons: bool = False
    debug_anomalies: bool = False


class TestBuildOccurrenceFromMatch:
    def test_strip_mode_does_not_include_trailing_dot(self, _patch):
        calls = {}

        def fake_normalize_acronym_key(base, allow_chars, dotted_mode):
            calls["normalize_acronym_key"] = (base, allow_chars, dotted_mode)
            return f"NK[{base}|{dotted_mode}]"

        def fake_context_window(text, s, e, win):
            calls["context_window"] = (text, s, e, win)
            return (111, 222)

        _patch(
            _build_occurrence_from_match,
            normalize_acronym_key=fake_normalize_acronym_key,
            context_window=fake_context_window,
        )

        cfg = _TestCfg(dotted_display="strip")
        text = "NASA."
        surface = "NASA"
        s, e = 0, 4  # '.' is at index 4

        occ, display_key = _build_occurrence_from_match(cfg, text, surface, s, e, conf=0.87)

        # display key returned from our fake normalizer
        assert display_key == "NK[NASA|strip]"

        # Occurrence fields
        assert isinstance(occ, Occurrence)
        assert occ.acronym == "NASA"
        assert occ.start_offset == 0
        assert occ.end_offset == 4
        assert occ.confidence == 0.87
        assert occ.context_window == (111, 222)
        assert occ.normalized_key == display_key
        assert occ.reasons is None

        assert calls["normalize_acronym_key"] == ("NASA", cfg.allow_chars, "strip")
        assert calls["context_window"] == (text, s, 4, cfg.window_chars)

    def test_preserve_mode_includes_trailing_dot(self, _patch):
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
            _build_occurrence_from_match,
            normalize_acronym_key=fake_normalize_acronym_key,
            context_window=fake_context_window,
        )

        cfg = _TestCfg(dotted_display="preserve")
        text = "N.A.S.A."
        surface = "N.A.S.A"
        s, e = 0, 4  # '.' at index 4

        occ, display_key = _build_occurrence_from_match(cfg, text, surface, s, e, conf=0.9)

        assert display_key == "NK[N.A.S.A|preserve]"
        assert occ.acronym == "N.A.S.A"  # dots INCLUDED
        assert occ.end_offset == 4  # advanced
        assert occ.context_window == (5, 10)

        # Helper call args used the adjusted end offset
        assert calls["normalize_acronym_key"] == ("N.A.S.A", cfg.allow_chars, "preserve")
        assert calls["context_window"] == (text, s, 4, cfg.window_chars)

    def test_debug_reasons_attached_when_enabled(self, _patch):
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
            _build_occurrence_from_match,
            normalize_acronym_key=fake_normalize_acronym_key,
            context_window=fake_context_window,
            reason_tags=fake_reason_tags,
        )

        cfg = _TestCfg(dotted_display="preserve", debug_reasons=True)
        text = "N.A.S.A."
        s = 0
        surface = "N.A.S.A"
        e = s + len(surface)  # 7, so text[e] is the trailing '.'
        assert cfg.dotted_display == "preserve"
        assert text[e] == ".", (e, text, text[e - 2:e + 2])

        occ, display_key = _build_occurrence_from_match(cfg, text, surface, s, e, conf=0.7)

        assert display_key == "NK"
        assert occ.reasons == ("EDGE", "PUNCT")
        assert occ.acronym == "N.A.S.A"
        assert occ.end_offset == e + 1  # == 8

    def test_returns_occurrence_and_display_key_tuple(self, _patch):

        cfg = _TestCfg(dotted_display="strip")
        text = "API."
        surface = "API"
        s, e = 0, 3
        _patch(
            _build_occurrence_from_match,
            normalize_acronym_key=lambda base, allow_chars, dotted_mode=None: f"{base.lower()}::{dotted_mode}",
            context_window=lambda text, s, e, w: (1, 2)
        )

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
        # normalize_acronym_key in strip mode removes dots; no dots present anyway, key equals base
        assert key_s == "GPU"

        # preserve mode → removes trailing dot in surface & end_offset
        cfg_pres = DetectorConfig(dotted_display="preserve")
        occ_p, key_p = _build_occurrence_from_match(
            cfg_pres, text, text[s_gpu:e_gpu], s_gpu, e_gpu, conf=0.91
        )
        assert occ_p.acronym == "GPU"  # dot not included
        assert occ_p.end_offset == e_gpu + 1
        assert occ_p.normalized_key == key_p
        # In preserve mode, key keeps the dot
        assert not key_p.endswith(".")

        # Context window invariants (don’t assert exact indices across implementations)
        l_s, r_s = occ_s.context_window
        l_p, r_p = occ_p.context_window
        n = len(text)
        for left, right, strt, end in [(l_s, r_s, occ_s.start_offset, occ_s.end_offset),
                                       (l_p, r_p, occ_p.start_offset, occ_p.end_offset)]:
            assert 0 <= left < right <= n
            assert left <= strt < end <= right

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


class TestScoreChunkWorkerUnit:
    def test_drops_blacklisted_candidates(self,  _patch):
        """
        If blacklist_context_drop(...) returns True, the candidate must be skipped.
        """
        cfg = _TestCfg()

        # Candidate set: first blacklisted, second allowed
        cands = [("GPU", 0, 3), ("API", 5, 8)]

        calls = {"build": []}

        # Stub: first candidate blacklisted only
        def fake_blacklist(surface, text, s, e, cfg_):
            return surface == "GPU"

        # Stub: give everyone a high score
        def fake_score(surface, text, s, e, cfg_):
            return 0.99

        # Stub: effective length (doesn't matter here)
        def fake_threshold_len(surface, allow_chars):
            return 3

        # Stub: build a minimal Occurrence
        def fake_build(cfg_, text, surface, s, e, conf):
            occ = Occurrence(
                acronym=surface,
                start_offset=s,
                end_offset=e,
                confidence=conf,
                context_window=(0, 0),
                normalized_key=surface,
                reasons=None,
            )
            calls["build"].append(surface)
            return occ, surface
        _patch(_score_chunk_worker,
               blacklist_context_drop=fake_blacklist,
               score=fake_score,
               threshold_len=fake_threshold_len,
               _build_occurrence_from_match=fake_build)

        out = _score_chunk_worker(cfg, text="GPU and API", cands=cands)
        # Only "API" should pass through
        assert [o.acronym for o in out] == ["API"]
        assert calls["build"] == ["API"]

    def test_threshold_gate_by_effective_length(self, _patch):
        """
        Conf < threshold → drop; conf == threshold → keep.
        Ensure the threshold chosen comes from threshold_len(surface,...).
        """
        cfg = _TestCfg()

        cands = [("AI", 0, 2), ("R&D", 4, 7)]  # "AI" eff=2, "R&D" eff>=3

        # Map per-surface scores
        scores = {"AI": 0.71, "R&D": 0.60}  # AI below 0.72 → drop; R&D == 0.60 → keep

        def fake_blacklist(surface, text, s, e, cfg_):
            return False

        def fake_score(surface, text, s, e, cfg_):
            return scores[surface]

        def fake_threshold_len(surface, allow_chars):
            return 2 if surface == "AI" else 3

        def fake_build(cfg_, text, surface, s, e, conf):
            return Occurrence(surface, s, e, conf, (0, 0), surface, None), surface

        _patch(_score_chunk_worker,
               blacklist_context_drop=fake_blacklist,
               score=fake_score,
               threshold_len=fake_threshold_len,
               _build_occurrence_from_match=fake_build)

        out = _score_chunk_worker(cfg, text="AI & R&D", cands=cands)
        # Only "R&D" should survive at equality
        assert [o.acronym for o in out] == ["R&D"]
        assert out[0].confidence == pytest.approx(0.60)

    def test_order_preserved_and_confidence_propagated(self, _patch):
        """
        Accepted occurrences are appended in input order, and their confidence equals score(...).
        """
        cfg = _TestCfg()
        cands = [("A", 0, 1), ("B", 2, 3), ("C", 4, 5)]

        # Drop B via blacklist; A and C accepted with distinct scores
        def fake_blacklist(surface, *_):
            return surface == "B"

        def fake_score(surface, *_):
            return {"A": 0.8, "B": 0.1, "C": 0.9}[surface]

        def fake_threshold_len(surface, *_):
            return 3  # so threshold = 0.60 for all

        def fake_build(cfg_, text, surface, s, e, conf):
            return Occurrence(surface, s, e, conf, (0, 0), surface, None), surface

        _patch(_score_chunk_worker,
               blacklist_context_drop=fake_blacklist,
               score=fake_score,
               threshold_len=fake_threshold_len,
               _build_occurrence_from_match=fake_build)

        out = _score_chunk_worker(cfg, text="A B C", cands=cands)
        assert [o.acronym for o in out] == ["A", "C"]
        assert [o.confidence for o in out] == [pytest.approx(0.8), pytest.approx(0.9)]

    def test_filters_and_builds_occurrences_end_to_end(self):
        """
        End-to-end through the real helpers:
        - Drops known non-acronym 'OK' when followed by punctuation.
        - Accepts 'R&D' (separator bumps effective len to >=3 so threshold = 0.60).
        - Accepts dotted initialism 'N.A.S.A' and, in 'preserve' mode, includes trailing '.'.
        """
        text = "OK, the R&D team met N.A.S.A. scientists."
        #             012345678901234567890123456789012345678901
        #             0         1         2         3         4
        s_ok = text.index("OK")
        e_ok = s_ok + 2
        s_rnd = text.index("R&D")
        e_rnd = s_rnd + 3
        s_nasa = text.index("N.A.S.A")
        e_nasa = s_nasa + len("N.A.S.A")  # '.' at e_nasa

        cfg = DetectorConfig(
            dotted_display="preserve",
            enable_dotted=True,
            allow_chars="&/-",  # allows '&' for 'R&D'
            window_chars=80,
        )

        cands = [
            ("OK", s_ok, e_ok),  # should be dropped
            ("R&D", s_rnd, e_rnd),  # should be accepted
            ("N.A.S.A", s_nasa, e_nasa),  # should be accepted; dot preserved
        ]

        out = _score_chunk_worker(cfg, text, cands)

        # Expect only 'R&D' and 'N.A.S.A.' (order preserved)
        assert [o.acronym for o in out] == ["R&D", "N.A.S.A"]
        assert all(isinstance(o, Occurrence) for o in out)

        # 'R&D' — no trailing dot added; end offset unchanged
        rnd = out[0]
        assert rnd.start_offset == s_rnd
        assert rnd.end_offset == e_rnd
        assert rnd.confidence >= 0.60  # score() baseline hits threshold

        # 'N.A.S.A' — preserve mode includes trailing '.' so end offset advances by 1
        nasa = out[1]
        assert nasa.start_offset == s_nasa
        assert nasa.end_offset == e_nasa + 1
        # Confidence should be >= threshold (effective len >= 3 → 0.60)
        assert nasa.confidence >= 0.60

        # Context-window sanity (bounds and containment)
        n = len(text)
        for o in out:
            left, right = o.context_window
            assert 0 <= left < right <= n
            assert left <= o.start_offset < o.end_offset <= right

    def test_builder_exception_skips_when_not_strict_and_logs(self, _patch):
        calls = {"events": []}  # define before fake_logger closes over it

        def fake_build(cfg_, text, surface, s, e, conf):
            if surface == "GPU":
                raise det.OccurrenceBuildError("synthetic build error")
            return det.Occurrence(surface, s, e, conf, (0, 0), surface, None), surface

        def fake_logger(event, **kw):
            calls["events"].append((event, kw))

        cfg = _TestCfg(debug_anomalies=True)
        cands = [("GPU", 0, 3), ("API", 5, 8)]

        _patch(
            _score_chunk_worker,
            blacklist_context_drop=lambda *a, **k: False,
            score=lambda *a, **k: 0.99,
            threshold_len=lambda *a, **k: 3,
            _build_occurrence_from_match=fake_build,
            message_logger=fake_logger,
        )

        out = _score_chunk_worker(cfg, text="GPU and API", cands=cands)

        assert [o.acronym for o in out] == ["API"]
        assert any(evt == "detector.bad_occurrence" for (evt, _kw) in calls["events"])

    def test_builder_exception_is_skipped(self, _patch):
        cfg = _TestCfg()  # debug_anomalies=False
        cands = [("GPU", 0, 3), ("API", 5, 8)]

        def fake_build(cfg_, text, surface, s, e, conf):
            if surface == "GPU":
                raise det.OccurrenceBuildError("boom")
            return det.Occurrence(surface, s, e, conf, (0, 0), surface, None), surface

        _patch(
            _score_chunk_worker,
            blacklist_context_drop=lambda *a, **k: False,
            score=lambda *a, **k: 0.99,
            threshold_len=lambda *a, **k: 3,
            _build_occurrence_from_match=fake_build,
        )

        out = _score_chunk_worker(cfg, text="GPU and API", cands=cands)
        assert [o.acronym for o in out] == ["API"]

    def test_builder_exception_logs_when_debug_anomalies_on(self, _patch):
        cfg = _TestCfg(debug_anomalies=True)
        cands = [("GPU", 0, 3), ("API", 5, 8)]
        events: list[str] = []

        def fake_build(cfg_, text, surface, s, e, conf):
            if surface == "GPU":
                raise det.OccurrenceBuildError("boom")
            return det.Occurrence("API", 5, 8, conf, (0, 0), "API", None), "API"

        _patch(
            _score_chunk_worker,
            blacklist_context_drop=lambda *a, **k: False,
            score=lambda *a, **k: 0.99,
            threshold_len=lambda *a, **k: 3,
            _build_occurrence_from_match=fake_build,
            message_logger=lambda event, **kw: events.append(event),
        )

        out = _score_chunk_worker(cfg, text="GPU and API", cands=cands)
        assert [o.acronym for o in out] == ["API"]
        assert "detector.bad_occurrence" in events

    def test_builder_exception_does_not_log_when_flag_off(self, _patch):
        cfg = _TestCfg()  # debug_anomalies False
        cands = [("GPU", 0, 3)]

        events: list[str] = []

        def fake_build(*_a, **_k):
            raise det.OccurrenceBuildError("boom")

        _patch(
            _score_chunk_worker,
            blacklist_context_drop=lambda *a, **k: False,
            score=lambda *a, **k: 0.99,
            threshold_len=lambda *a, **k: 3,
            _build_occurrence_from_match=fake_build,
            message_logger=lambda event, **kw: events.append(event),
        )

        out = _score_chunk_worker(cfg, text="GPU", cands=cands)
        assert out == []
        assert events == []  # no log when debug_anomalies is off
