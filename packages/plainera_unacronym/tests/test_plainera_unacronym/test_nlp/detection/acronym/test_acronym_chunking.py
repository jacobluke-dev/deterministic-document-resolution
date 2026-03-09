import plainera_unacronym.nlp.detection.acronym.detector as det
import pytest
from plainera_unacronym.nlp import Occurrence
from plainera_unacronym.nlp.detection.acronym.chunking import score_chunk_worker


class TestScoreChunkWorkerUnit:
    def test_drops_blacklisted_candidates(self, _patch, test_cfg):
        """
        If blacklist_context_drop(...) returns True, the candidate must be skipped.
        """
        cfg = test_cfg()

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
                occurrence_confidence=conf,
                segment_window=(0, 0),
                normalized_key=surface,
                reasons=None,
            )
            calls["build"].append(surface)
            return occ, surface

        _patch(
            score_chunk_worker,
            blacklist_context_drop=fake_blacklist,
            score=fake_score,
            threshold_len=fake_threshold_len,
            build_occurrence_from_match=fake_build,
        )

        out = score_chunk_worker(cfg, text="GPU and API", cands=cands)
        # Only "API" should pass through
        assert [o.acronym for o in out] == ["API"]
        assert calls["build"] == ["API"]

    def test_threshold_gate_by_effective_length(self, _patch, test_cfg):
        """
        Conf < threshold → drop; conf == threshold → keep.
        Ensure the threshold chosen comes from threshold_len(surface,...).
        """
        cfg = test_cfg()

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

        _patch(
            score_chunk_worker,
            blacklist_context_drop=fake_blacklist,
            score=fake_score,
            threshold_len=fake_threshold_len,
            build_occurrence_from_match=fake_build,
        )

        out = score_chunk_worker(cfg, text="AI & R&D", cands=cands)
        # Only "R&D" should survive at equality
        assert [o.acronym for o in out] == ["R&D"]
        assert out[0].occurrence_confidence == pytest.approx(0.60)

    def test_order_preserved_and_confidence_propagated(self, _patch, test_cfg):
        """
        Accepted occurrences are appended in input order, and their confidence equals score(...).
        """
        cfg = test_cfg()
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

        _patch(
            score_chunk_worker,
            blacklist_context_drop=fake_blacklist,
            calc_score=fake_score,
            threshold_len=fake_threshold_len,
            build_occurrence_from_match=fake_build,
        )

        out = score_chunk_worker(cfg, text="A B C", cands=cands)
        assert [o.acronym for o in out] == ["A", "C"]
        assert [o.occurrence_confidence for o in out] == [pytest.approx(0.8), pytest.approx(0.9)]

    def test_filters_and_builds_occurrences_end_to_end(self, test_cfg):
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

        cfg = test_cfg(
            dotted_display="preserve",
            enable_dotted=True,
        )

        cands = [
            ("OK", s_ok, e_ok),  # should be dropped
            ("R&D", s_rnd, e_rnd),  # should be accepted
            ("N.A.S.A", s_nasa, e_nasa),  # should be accepted; dot preserved
        ]

        out = score_chunk_worker(cfg, text, cands)

        # Expect only 'R&D' and 'N.A.S.A.' (order preserved)
        assert [o.acronym for o in out] == ["R&D", "N.A.S.A"]
        assert all(isinstance(o, Occurrence) for o in out)

        # 'R&D' — no trailing dot added; end offset unchanged
        rnd = out[0]
        assert rnd.start_offset == s_rnd
        assert rnd.end_offset == e_rnd
        assert rnd.occurrence_confidence >= 0.60  # score() baseline hits threshold

        # 'N.A.S.A' — preserve mode includes trailing '.' so end offset advances by 1
        nasa = out[1]
        assert nasa.start_offset == s_nasa
        assert nasa.end_offset == e_nasa + 1
        # Confidence should be >= threshold (effective len >= 3 → 0.60)
        assert nasa.occurrence_confidence >= 0.60

        # Context-window sanity (bounds and containment)
        n = len(text)
        for o in out:
            left, right = o.segment_window
            assert 0 <= left < right <= n
            assert left <= o.start_offset < o.end_offset <= right

    def test_builder_exception_skips_when_not_strict_and_logs(self, _patch, test_cfg):
        calls = {"events": []}  # define before fake_logger closes over it

        def fake_build(cfg_, text, surface, s, e, conf):
            if surface == "GPU":
                raise det.OccurrenceBuildError("synthetic build error")
            return det.Occurrence(surface, s, e, conf, (0, 0), surface, None), surface

        def fake_logger(event, **kw):
            calls["events"].append((event, kw))

        cfg = test_cfg(debug_anomalies=True)
        cands = [("GPU", 0, 3), ("API", 5, 8)]

        _patch(
            score_chunk_worker,
            blacklist_context_drop=lambda *a, **k: False,
            score=lambda *a, **k: 0.99,
            threshold_len=lambda *a, **k: 3,
            build_occurrence_from_match=fake_build,
            message_logger=fake_logger,
        )

        out = score_chunk_worker(cfg, text="GPU and API", cands=cands)

        assert [o.acronym for o in out] == ["API"]
        assert any(evt == "detector.bad_occurrence" for (evt, _kw) in calls["events"])

    def test_builder_exception_is_skipped(self, _patch, test_cfg):
        cfg = test_cfg()  # debug_anomalies=False
        cands = [("GPU", 0, 3), ("API", 5, 8)]

        def fake_build(cfg_, text, surface, s, e, conf):
            if surface == "GPU":
                raise det.OccurrenceBuildError("boom")
            return det.Occurrence(surface, s, e, conf, (0, 0), surface, None), surface

        _patch(
            score_chunk_worker,
            blacklist_context_drop=lambda *a, **k: False,
            score=lambda *a, **k: 0.99,
            threshold_len=lambda *a, **k: 3,
            build_occurrence_from_match=fake_build,
        )

        out = score_chunk_worker(cfg, text="GPU and API", cands=cands)
        assert [o.acronym for o in out] == ["API"]

    def test_builder_exception_logs_when_debug_anomalies_on(self, _patch, test_cfg):
        cfg = test_cfg(debug_anomalies=True)
        cands = [("GPU", 0, 3), ("API", 5, 8)]
        events: list[str] = []

        def fake_build(cfg_, text, surface, s, e, conf):
            if surface == "GPU":
                raise det.OccurrenceBuildError("boom")
            return det.Occurrence("API", 5, 8, conf, (0, 0), "API", None), "API"

        _patch(
            score_chunk_worker,
            blacklist_context_drop=lambda *a, **k: False,
            score=lambda *a, **k: 0.99,
            threshold_len=lambda *a, **k: 3,
            build_occurrence_from_match=fake_build,
            message_logger=lambda event, **kw: events.append(event),
        )

        out = score_chunk_worker(cfg, text="GPU and API", cands=cands)
        assert [o.acronym for o in out] == ["API"]
        assert "detector.bad_occurrence" in events

    def test_builder_exception_does_not_log_when_flag_off(self, _patch, test_cfg):
        cfg = test_cfg()  # debug_anomalies False
        cands = [("GPU", 0, 3)]

        events: list[str] = []

        def fake_build(*_a, **_k):
            raise det.OccurrenceBuildError("boom")

        _patch(
            score_chunk_worker,
            blacklist_context_drop=lambda *a, **k: False,
            score=lambda *a, **k: 0.99,
            threshold_len=lambda *a, **k: 3,
            build_occurrence_from_match=fake_build,
            message_logger=lambda event, **kw: events.append(event),
        )

        out = score_chunk_worker(cfg, text="GPU", cands=cands)
        assert out == []
        assert events == []  # no log when debug_anomalies is off
