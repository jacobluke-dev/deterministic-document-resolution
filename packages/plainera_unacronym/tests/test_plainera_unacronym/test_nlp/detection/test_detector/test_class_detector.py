import asyncio
from dataclasses import replace
from types import SimpleNamespace

import plainera_unacronym.nlp.detection.detector as det
import pytest
from plainera_unacronym.nlp.detection.detector import (
    Detector,
    DetectorConfig,
    Occurrence,
)

# ----- helpers -----

@pytest.fixture
def cfg_factory():
    def make(**overrides) -> DetectorConfig:
        return replace(DetectorConfig(), **overrides)
    return make


class _SpyLog:
    def __init__(self):
        self.calls = []

    def __call__(self, message, *_, **kw):
        self.calls.append({"message": message, **kw})


def _occ(acr: str, s: int, e: int, conf: float, key: str | None = None) -> Occurrence:
    return Occurrence(
        acronym=acr,
        start_offset=s,
        end_offset=e,
        confidence=conf,
        context_window=(0, 0),
        normalized_key=key,
        reasons=None,
    )


class TestDetectorUnit:
    def test__with_auto_domains_merges_and_short_circuits(self, cfg_factory, monkeypatch):
        cfg0 = cfg_factory(enabled_domains=frozenset({"bio"}))
        d = Detector(cfg0, max_workers=1)

        # Case 1: adds new domain → returns REPLACED config
        monkeypatch.setattr(det, "autodetect_domains", lambda text, cfg: frozenset({"finance", "bio"}), raising=True)
        out = d._with_auto_domains("some text mentioning markets")
        assert out is not d.cfg
        assert out.enabled_domains == frozenset({"bio", "finance"})

        # Case 2: nothing new → returns the same object (identity)
        monkeypatch.setattr(det, "autodetect_domains", lambda text, cfg: frozenset({"bio"}), raising=True)
        out2 = d._with_auto_domains("no change")
        assert out2 is d.cfg

    def test_detect_counts_firsts_and_logs(self, cfg_factory, monkeypatch):
        cfg = cfg_factory(dotted_display="strip", allow_chars="&/-")
        d = Detector(cfg)

        # Patch candidate iteration and scoring/threshold pipeline
        cands = [("GPU", 0, 3), ("GPU", 10, 13), ("API", 20, 23), ("OK", 30, 32)]
        monkeypatch.setattr(det, "compile_pattern", lambda _cfg: object(), raising=True)
        monkeypatch.setattr(det, "iter_candidates_with", lambda text, cfg, pat: cands, raising=True)
        monkeypatch.setattr(det, "blacklist_context_drop", lambda surf, *_: (surf == "OK"), raising=True)
        scores = {"GPU": 0.91, "API": 0.90, "OK": 0.99}
        monkeypatch.setattr(det, "score", lambda surf, *_: scores[surf], raising=True)
        monkeypatch.setattr(det, "threshold_len", lambda surf, ac: 3, raising=True)  # -> threshold = 0.60
        # Build occurrences; normalized key = surface (simple)
        monkeypatch.setattr(
            det,
            "_build_occurrence_from_match",
            lambda cfg_, text, sfc, s, e, conf: (_occ(sfc, s, e, conf, key=sfc), sfc),
            raising=True,
        )

        spy = _SpyLog()
        monkeypatch.setattr(det, "message_logger", spy, raising=True)
        # Avoid domain auto-add noise
        monkeypatch.setattr(det, "autodetect_domains", lambda text, cfg_: frozenset(), raising=True)

        res = d.detect("GPU then GPU, and API; OK should drop.")
        # Accepted: 3 (two GPU + one API); unique: 2
        assert [o.acronym for o in res.occurrences] == ["GPU", "GPU", "API"]
        assert set(res.unique_acronyms.keys()) == {"GPU", "API"}
        # Logs emitted
        kinds = [c["message"] for c in spy.calls]
        assert "detector.detect.start" in kinds
        assert "detector.detect.summary" in kinds
        summary = next(c for c in spy.calls if c["message"] == "detector.detect.summary")
        assert summary["args"]["candidates"] == len(cands)
        assert summary["args"]["dropped_blacklist"] == 1
        assert summary["args"]["accepted"] == 3
        assert summary["args"]["unique"] == 2
        assert isinstance(summary["details"]["top"], list)

    def test_detect_respects_thresholds(self, cfg_factory, monkeypatch):
        cfg = cfg_factory()
        d = Detector(cfg)

        cands = [("AI", 0, 2), ("R&D", 5, 8)]
        monkeypatch.setattr(det, "compile_pattern", lambda _: object(), raising=True)
        monkeypatch.setattr(det, "iter_candidates_with", lambda *_: cands, raising=True)
        monkeypatch.setattr(det, "blacklist_context_drop", lambda *_: False, raising=True)
        # AI below 2-letter threshold (0.72); R&D at 3-letter threshold (0.60)
        sc = {"AI": 0.71, "R&D": 0.60}
        monkeypatch.setattr(det, "score", lambda surf, *_: sc[surf], raising=True)
        monkeypatch.setattr(det, "threshold_len", lambda surf, ac: 2 if surf == "AI" else 3, raising=True)
        monkeypatch.setattr(
            det, "_build_occurrence_from_match", lambda *_: (_occ("R&D", 5, 8, 0.60, "R&D"), "R&D"), raising=True
        )
        monkeypatch.setattr(det, "message_logger", lambda *a, **k: None, raising=True)
        monkeypatch.setattr(det, "autodetect_domains", lambda *_: frozenset(), raising=True)

        res = d.detect("AI and R&D")
        assert [o.acronym for o in res.occurrences] == ["R&D"]
        assert set(res.unique_acronyms.keys()) == {"R&D"}

    def test_detect_parallel_defers_to_serial_when_below_threshold(self, cfg_factory, monkeypatch):
        cfg = cfg_factory()
        d = Detector(cfg)

        monkeypatch.setattr(det, "compile_pattern", lambda _: object(), raising=True)
        monkeypatch.setattr(det, "iter_candidates_with", lambda *_: [("GPU", 0, 3)], raising=True)

        called = {"detect": 0}
        def fake_detect(self, text):
            called["detect"] += 1
            return SimpleNamespace(unique_acronyms={}, occurrences=[])

        monkeypatch.setattr(Detector, "detect", fake_detect, raising=True)
        res = d.detect_parallel("short", threshold=10, chunk_size=256)
        assert called["detect"] == 1
        assert hasattr(res, "unique_acronyms") and hasattr(res, "occurrences")

    def test_detect_parallel_merges_chunks_builds_firsts_and_handles_missing_key(self, cfg_factory, monkeypatch):
        cfg = cfg_factory(dotted_display="strip", allow_chars="&/-")
        d = Detector(cfg)

        # Create enough candidates for 2 chunks
        cands = [("GPU", 0, 3), ("API", 5, 8), ("GPU", 10, 13)]
        monkeypatch.setattr(det, "compile_pattern", lambda _: object(), raising=True)
        monkeypatch.setattr(det, "iter_candidates_with", lambda *_: cands, raising=True)

        # Fake pool that executes synchronously
        class FakeFuture:
            def __init__(self, fn, args): self._fn, self._args = fn, args
            def result(self): return self._fn(*self._args)

        class FakePool:
            def __init__(self): self.submitted = []
            def submit(self, fn, *args):
                self.submitted.append((fn, args))
                return FakeFuture(fn, args)

        monkeypatch.setattr(det, "ProcessPoolExecutor", lambda **kw: FakePool(), raising=True)

        # Worker returns Occurrences; first one with missing normalized_key to test fallback
        def fake_worker(cfg_, text, chunk):
            outs = []
            for sfc, s, e in chunk:
                key = sfc if sfc != "GPU" else None  # force missing key for GPU
                outs.append(_occ(sfc, s, e, 0.9, key))
            return outs

        monkeypatch.setattr(det, "_score_chunk_worker", fake_worker, raising=True)
        # normalize_key used for fallback
        monkeypatch.setattr(det, "normalize_acronym_key", lambda acr, allow, dotted_mode=None: f"N[{acr}]", raising=True)
        monkeypatch.setattr(det, "message_logger", lambda *a, **k: None, raising=True)
        monkeypatch.setattr(det, "autodetect_domains", lambda *_: frozenset(), raising=True)

        res = d.detect_parallel("x" * 2000, threshold=1, chunk_size=2)
        # Firsts should include fallback-normalized GPU and API
        assert set(res.unique_acronyms.keys()) == {"N[GPU]", "API"}
        # Occurrences preserved
        assert [o.acronym for o in res.occurrences] == ["GPU", "API", "GPU"]

    def test_parallel_fallback_when_key_is_none(self, cfg_factory, monkeypatch):
        cfg = cfg_factory()
        d = Detector(cfg)

        monkeypatch.setattr(det, "compile_pattern", lambda _: object(), raising=True)
        monkeypatch.setattr(det, "iter_candidates_with", lambda *_: [("GPU", 0, 3)], raising=True)

        class FakeFuture:
            def __init__(self, fn, args): self.fn, self.args = fn, args

            def result(self): return self.fn(*self.args)

        class FakePool:
            def submit(self, fn, *args): return FakeFuture(fn, args)

        monkeypatch.setattr(det, "ProcessPoolExecutor", lambda **kw: FakePool(), raising=True)

        # Worker returns an Occurrence with normalized_key=None
        monkeypatch.setattr(det, "_score_chunk_worker",
                            lambda *_: [Occurrence("GPU", 0, 3, 0.9, (0, 0), None, None)],  # key=None
                            raising=True
                            )
        monkeypatch.setattr(det, "normalize_acronym_key",
                            lambda acr, allow, dotted_mode=None: f"N[{acr}]",
                            raising=True
                            )
        monkeypatch.setattr(det, "message_logger", lambda *a, **k: None, raising=True)
        monkeypatch.setattr(det, "autodetect_domains", lambda *_: frozenset(), raising=True)

        res = d.detect_parallel("x" * 2000, threshold=1, chunk_size=1)
        assert set(res.unique_acronyms.keys()) == {"N[GPU]"}

    def test_detect_parallel_logs_chunk_failure_and_continues(self, cfg_factory, monkeypatch):
        cfg = cfg_factory()
        d = Detector(cfg)

        cands = [("A", 0, 1), ("B", 2, 3), ("C", 4, 5)]
        monkeypatch.setattr(det, "compile_pattern", lambda _: object(), raising=True)
        monkeypatch.setattr(det, "iter_candidates_with", lambda *_: cands, raising=True)

        class FakeFuture:
            def __init__(self, fn, args, should_fail=False):
                self.fn, self.args, self.should_fail = fn, args, should_fail
            def result(self):
                if self.should_fail:
                    raise RuntimeError("boom")
                return self.fn(*self.args)

        class FakePool:
            def __init__(self): self.i = 0
            def submit(self, fn, *args):
                self.i += 1
                return FakeFuture(fn, args, should_fail=(self.i == 2))

        monkeypatch.setattr(det, "ProcessPoolExecutor", lambda **kw: FakePool(), raising=True)
        monkeypatch.setattr(det, "_score_chunk_worker", lambda *_: [_occ("OK", 0, 1, 0.6, "OK")], raising=True)
        logs = _SpyLog()
        monkeypatch.setattr(det, "message_logger", logs, raising=True)
        monkeypatch.setattr(det, "autodetect_domains", lambda *_: frozenset(), raising=True)

        res = d.detect_parallel("X" * 5000, threshold=1, chunk_size=1)
        # We still get occurrences from the chunks that didn't fail
        assert any(o.acronym == "OK" for o in res.occurrences)
        # Failure was logged
        assert any(c["message"] == "detector.chunk.failed" for c in logs.calls)

    @pytest.mark.asyncio
    async def test_detect_async_delegates_to_thread(self, cfg_factory, monkeypatch):
        cfg = cfg_factory()
        d = Detector(cfg)

        # Make to_thread run synchronously for the test
        async def fake_to_thread(fn, *a, **k):
            return fn(*a, **k)

        monkeypatch.setattr(asyncio, "to_thread", fake_to_thread, raising=True)
        # Avoid heavy deps
        monkeypatch.setattr(det, "compile_pattern", lambda _: object(), raising=True)
        monkeypatch.setattr(det, "iter_candidates_with", lambda *_: [], raising=True)
        monkeypatch.setattr(det, "message_logger", lambda *a, **k: None, raising=True)
        monkeypatch.setattr(det, "autodetect_domains", lambda *_: frozenset(), raising=True)

        res = await d.detect_async("nothing here")
        assert hasattr(res, "unique_acronyms") and hasattr(res, "occurrences")


class TestDetectorIntegration:
    def test_end_to_end_common_cases(self):
        cfg = DetectorConfig(dotted_display="strip", enable_dotted=False)
        d = Detector(cfg)

        text = "We’ll loop in R & D after the NHS workshop. OK, let's meet at 10:30 AM."
        res = d.detect(text)

        ks = set(res.unique_acronyms.keys())
        # normalized separators and core tokens
        assert "R&D" in ks and "NHS" in ks
        # noisy known uppers / time tokens drop
        assert "OK" not in ks and "AM" not in ks

    def test_dotted_mode_toggle_affects_US_UK(self):
        txt = "The U.S. economy and U.K. policy differ. NASA leads."

        # OFF (default): dotted forms not matched
        off = Detector(DetectorConfig(enable_dotted=False, dotted_display="strip")).detect(txt)
        assert "US" not in off.unique_acronyms and "UK" not in off.unique_acronyms
        assert "NASA" in off.unique_acronyms

        # ON: dotted initialisms accepted; dots stripped for key in strip mode
        on = Detector(DetectorConfig(enable_dotted=True, dotted_display="strip")).detect(txt)
        assert "US" in on.unique_acronyms and "UK" in on.unique_acronyms

    def test_dotted_initialisms_are_preserved_in_keys(self):
        txt = "The U.S. economy and U.K. policy differ. NASA leads."
        cfg = DetectorConfig(enable_dotted=True, dotted_display="preserve")
        res = Detector(cfg).detect(txt)

        keys = set(res.unique_acronyms.keys())
        print("KEYS ARE ... ", keys)
        # With dotted_display='preserve' we expect dots in the keys:
        assert "U.S" in keys, f"Missing 'U.S.' in keys: {keys}"
        assert "U.K" in keys, f"Missing 'U.K.' in keys: {keys}"
        # Sanity: other acronyms still appear
        assert "NASA" in keys, f"Missing 'NASA' in keys: {keys}"

    def test_dotted_initialisms_strip_mode_normalizes_without_dots(self):
        txt = "The U.S. economy and U.K. policy differ. NASA leads."
        cfg = DetectorConfig(enable_dotted=True, dotted_display="strip")
        res = Detector(cfg).detect(txt)

        keys = set(res.unique_acronyms.keys())
        assert "US" in keys and "UK" in keys, f"Expected 'US' and 'UK' when stripping dots, got: {keys}"
        assert "U.S." not in keys and "U.K." not in keys
        assert "NASA" in keys

    def test_mixed_case_toggle_affects_tfl(self):
        txt = "Transport for London (TfL) runs the Tube. TfL operates buses."

        # Mixed-case enabled → should keep the original casing key "TfL"
        mc_on = Detector(DetectorConfig(enable_mixed_case=True)).detect(txt)
        on_keys = set(mc_on.unique_acronyms.keys())
        assert "TfL" in on_keys, f"Expected 'TfL' with mixed-case enabled, got {on_keys}"

        # Mixed-case disabled → should not surface TfL (nor an uppercased TFL)
        mc_off = Detector(DetectorConfig(enable_mixed_case=False)).detect(txt)
        off_keys = set(mc_off.unique_acronyms.keys())
        assert "TFL" not in off_keys, f"Did not expect 'TFL' with mixed-case disabled, got {off_keys}"
        assert "TfL" not in off_keys, f"Did not expect 'TfL' with mixed-case disabled, got {off_keys}"
