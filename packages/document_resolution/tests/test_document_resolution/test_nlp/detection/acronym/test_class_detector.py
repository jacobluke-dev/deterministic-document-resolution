from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import document_resolution.nlp.detection.acronym.detector as det
import document_resolution.nlp.detection.base as bs
import pytest
from document_resolution.nlp.common.types import AcronymDetectorConfig, Occurrence
from document_resolution.nlp.detection.acronym.detector import AcronymDetector

# ----- helpers -----


@pytest.fixture
def cfg_factory():
    def make(**overrides) -> AcronymDetectorConfig:
        return replace(AcronymDetectorConfig(), **overrides)

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
        occurrence_confidence=conf,
        segment_window=(0, 0),
        normalized_key=key,
        reasons=None,
    )


class TestAcronymDetectorUnit:
    def test__with_auto_domains_merges_and_short_circuits(self, cfg_factory, _patch):
        cfg0 = cfg_factory(enabled_domains=frozenset({"bio"}))
        d = AcronymDetector(cfg0, max_workers=1)

        # Case 1: adds new domain → returns REPLACED config
        _patch(AcronymDetector._with_auto_domains, autodetect_domains=lambda text, cfg: frozenset({"finance", "bio"}))
        out = d._with_auto_domains("some text mentioning markets")
        assert out is not d.cfg
        assert out.enabled_domains == frozenset({"bio", "finance"})

        # Case 2: nothing new → returns the same object (identity)
        _patch(AcronymDetector._with_auto_domains, autodetect_domains=lambda text, cfg: frozenset({"bio"}))
        out2 = d._with_auto_domains("no change")
        assert out2 is d.cfg

    def test_detect_counts_firsts_and_logs(self, cfg_factory, _patch):
        cfg = cfg_factory(dotted_display="strip", allow_chars="&/-")
        d = AcronymDetector(cfg)

        detect = AcronymDetector.detect.__wrapped__  # type: ignore[attr-defined]

        cands = [("GPU", 0, 3), ("GPU", 10, 13), ("API", 20, 23), ("OK", 30, 32)]
        scores = {"GPU": 0.91, "API": 0.90, "OK": 0.99}
        spy = _SpyLog()

        _patch(
            detect,
            iter_acronym_candidates=lambda text, cfg, pat: cands,
            blacklist_context_drop=lambda surf, *_: surf == "OK",
            calc_score=lambda surf, *_: scores[surf],
            threshold_len=lambda surf, ac: 3,
            build_occurrence_from_match=lambda cfg_, text, sfc, s, e, conf: (
                _occ(sfc, s, e, conf, key=sfc),
                sfc,
            ),
            message_logger=spy,
        )
        _patch(AcronymDetector._with_auto_domains, autodetect_domains=lambda text, cfg_: frozenset())

        res = d.detect("GPU then GPU, and API; OK should drop.")

        assert [o.acronym for o in res.occurrences] == ["GPU", "GPU", "API"]
        assert set(res.unique_acronyms.keys()) == {"GPU", "API"}

        kinds = [c["message"] for c in spy.calls]
        assert "acronym_detector.detect.start" in kinds
        assert "acronym_detector.detect.summary" in kinds

        summary = next(c for c in spy.calls if c["message"] == "acronym_detector.detect.summary")
        assert summary["args"]["candidates"] == len(cands)
        assert summary["args"]["dropped_blacklist"] == 1
        assert summary["args"]["accepted"] == 3
        assert summary["args"]["unique"] == 2
        assert isinstance(summary["details"]["top"], list)

    def test_detect_respects_thresholds(self, cfg_factory, _patch):
        cfg = cfg_factory()
        d = AcronymDetector(cfg)

        detect = AcronymDetector.detect.__wrapped__  # type: ignore[attr-defined]

        cands = [("AI", 0, 2), ("R&D", 5, 8)]
        sc = {"AI": 0.71, "R&D": 0.60}

        _patch(
            detect,
            iter_acronym_candidates=lambda *_: cands,
            blacklist_context_drop=lambda *_: False,
            calc_score=lambda surf, *_: sc[surf],
            threshold_len=lambda surf, ac: 2 if surf == "AI" else 3,
            build_occurrence_from_match=lambda *_: (_occ("R&D", 5, 8, 0.60, "R&D"), "R&D"),
            message_logger=lambda *a, **k: None,
        )
        _patch(AcronymDetector._with_auto_domains, autodetect_domains=lambda *_: frozenset())

        res = d.detect("AI and R&D")

        assert [o.acronym for o in res.occurrences] == ["R&D"]
        assert set(res.unique_acronyms.keys()) == {"R&D"}


    def test_detect_parallel_defers_to_serial_when_below_threshold(self, cfg_factory, _patch):
        cfg = cfg_factory()
        d = AcronymDetector(cfg)

        detect_parallel = AcronymDetector.detect_parallel.__wrapped__  # type: ignore[attr-defined]
        _patch(detect_parallel, iter_acronym_candidates=lambda *_: [("GPU", 0, 3)])

        called = {"detect": 0}

        def fake_detect(text):
            called["detect"] += 1
            return SimpleNamespace(unique_acronyms={}, occurrences=[])

        d.detect = fake_detect  # type: ignore[method-assign]

        res = d.detect_parallel("short", threshold=10, chunk_size=256)

        assert called["detect"] == 1
        assert hasattr(res, "unique_acronyms") and hasattr(res, "occurrences")


    def test_detect_parallel_merges_chunks_builds_firsts_and_handles_missing_key(self, cfg_factory, _patch):
        cfg = cfg_factory(dotted_display="strip", allow_chars="&/-")
        d = AcronymDetector(cfg)

        # Create enough candidates for 2 chunks
        cands = [("GPU", 0, 3), ("API", 5, 8), ("GPU", 10, 13)]

        detect_parallel = AcronymDetector.detect_parallel.__wrapped__ # type: ignore[attr-defined]

        # Fake pool that executes synchronously
        class FakeFuture:
            def __init__(self, fn, args):
                self._fn, self._args = fn, args

            def result(self):
                return self._fn(*self._args)

        class FakePool:
            def __init__(self):
                self.submitted = []

            def submit(self, fn, *args):
                self.submitted.append((fn, args))
                return FakeFuture(fn, args)

        _patch(bs.BaseDetector._get_or_create_pool, ProcessPoolExecutor=lambda **kw: FakePool())

        # Worker returns Occurrences; first one with missing normalized_key to test fallback
        def fake_worker(cfg_, text, chunk):
            outs = []
            for sfc, s, e in chunk:
                key = sfc if sfc != "GPU" else None  # force missing key for GPU
                outs.append(_occ(sfc, s, e, 0.9, key))
            return outs

        _patch(
            detect_parallel,
            iter_acronym_candidates=lambda *_: cands,
            score_chunk_worker=fake_worker,
            normalize_acronym_key=lambda acr, allow, dotted_mode=None: f"N[{acr}]",
            message_logger=lambda *a, **k: None,
        )

        _patch(AcronymDetector._with_auto_domains, autodetect_domains=lambda *_: frozenset())

        res = d.detect_parallel("x" * 2000, threshold=1, chunk_size=2)
        # Firsts should include fallback-normalized GPU and API
        assert set(res.unique_acronyms.keys()) == {"N[GPU]", "API"}
        # Occurrences preserved
        assert [o.acronym for o in res.occurrences] == ["GPU", "API", "GPU"]

    def test_parallel_fallback_when_key_is_none(self, cfg_factory, _patch):
        cfg = cfg_factory()
        d = AcronymDetector(cfg)

        detect_parallel = AcronymDetector.detect_parallel.__wrapped__ # type: ignore[attr-defined]

        class FakeFuture:
            def __init__(self, fn, args):
                self.fn = fn
                self.args = args

            def result(self):
                return self.fn(*self.args)

        class FakePool:
            def submit(self, fn, *args):
                return FakeFuture(fn, args)

        _patch(
            detect_parallel,
            iter_acronym_candidates=lambda *_: [("GPU", 0, 3)],
            score_chunk_worker=lambda *_: [Occurrence("GPU", 0, 3, 0.9, (0, 0), None, None)],
            normalize_acronym_key=lambda acr, allow, dotted_mode=None: f"N[{acr}]",
            message_logger=lambda *a, **k: None,
        )
        _patch(bs.BaseDetector._get_or_create_pool, ProcessPoolExecutor=lambda **kw: FakePool())
        _patch(AcronymDetector._with_auto_domains, autodetect_domains=lambda *_: frozenset())

        res = d.detect_parallel("x" * 2000, threshold=1, chunk_size=1)

        assert set(res.unique_acronyms.keys()) == {"N[GPU]"}

    def test_detect_parallel_logs_chunk_failure_and_continues(self, cfg_factory, _patch):
        cfg = cfg_factory()
        d = AcronymDetector(cfg)

        detect_parallel = AcronymDetector.detect_parallel.__wrapped__  # type: ignore[attr-defined]

        class FakeFuture:
            def __init__(self, fn, args, should_fail=False):
                self.fn = fn
                self.args = args
                self.should_fail = should_fail

            def result(self):
                if self.should_fail:
                    raise RuntimeError("boom")
                return self.fn(*self.args)

        class FakePool:
            def __init__(self):
                self.i = 0

            def submit(self, fn, *args):
                self.i += 1
                return FakeFuture(fn, args, should_fail=(self.i == 2))

        logs = _SpyLog()

        cands = [("A", 0, 1), ("B", 2, 3), ("C", 4, 5)]

        _patch(
            detect_parallel,
            iter_acronym_candidates=lambda *_: cands,
            score_chunk_worker=lambda *_: [_occ("OK", 0, 1, 0.6, "OK")],
            message_logger=logs,
        )
        _patch(bs.BaseDetector._get_or_create_pool, ProcessPoolExecutor=lambda **kw: FakePool())
        _patch(AcronymDetector._with_auto_domains, autodetect_domains=lambda *_: frozenset())

        res = d.detect_parallel("X" * 5000, threshold=1, chunk_size=1)

        assert any(o.acronym == "OK" for o in res.occurrences)
        assert any(c["message"] == "acronym_detector.chunk.failed" for c in logs.calls)

    @pytest.mark.asyncio
    async def test_detect_async_delegates_to_thread(self, cfg_factory, _patch):
        cfg = cfg_factory()
        d = AcronymDetector(cfg)

        async def fake_to_thread(fn, *a, **k):
            return fn(*a, **k)

        detect = AcronymDetector.detect.__wrapped__  # type: ignore[attr-defined]

        _patch(
            bs.BaseDetector.detect_async,
            asyncio=SimpleNamespace(to_thread=fake_to_thread),
        )
        _patch(
            detect,
            iter_acronym_candidates=lambda *_: [],
            message_logger=lambda *a, **k: None,
        )
        _patch(AcronymDetector._with_auto_domains, autodetect_domains=lambda *_: frozenset())

        res = await d.detect_async("nothing here")

        assert hasattr(res, "unique_acronyms")
        assert hasattr(res, "occurrences")


class TestAcronymDetectorIntegration:
    def test_end_to_end_common_cases(self):
        cfg = AcronymDetectorConfig(dotted_display="strip", enable_dotted=False)
        d = AcronymDetector(cfg)

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
        off = AcronymDetector(AcronymDetectorConfig(enable_dotted=False, dotted_display="strip")).detect(txt)
        assert "US" not in off.unique_acronyms and "UK" not in off.unique_acronyms
        assert "NASA" in off.unique_acronyms

        # ON: dotted initialisms accepted; dots stripped for key in strip mode
        on = AcronymDetector(AcronymDetectorConfig(enable_dotted=True, dotted_display="strip")).detect(txt)
        assert "US" in on.unique_acronyms and "UK" in on.unique_acronyms

    def test_dotted_initials_are_preserved_in_keys(self):
        txt = "The U.S. economy and U.K. policy differ. NASA leads."
        cfg = AcronymDetectorConfig(enable_dotted=True, dotted_display="preserve")
        res = AcronymDetector(cfg).detect(txt)

        keys = set(res.unique_acronyms.keys())
        assert "U.S" in keys, f"Missing 'U.S.' in keys: {keys}"
        assert "U.K" in keys, f"Missing 'U.K.' in keys: {keys}"
        # Sanity: other acronyms still appear
        assert "NASA" in keys, f"Missing 'NASA' in keys: {keys}"

    def test_dotted_initialisms_strip_mode_normalizes_without_dots(self):
        txt = "The U.S. economy and U.K. policy differ. NASA leads."
        cfg = AcronymDetectorConfig(enable_dotted=True, dotted_display="strip")
        res = AcronymDetector(cfg).detect(txt)

        keys = set(res.unique_acronyms.keys())
        assert "US" in keys and "UK" in keys, f"Expected 'US' and 'UK' when stripping dots, got: {keys}"
        assert "U.S." not in keys and "U.K." not in keys
        assert "NASA" in keys

    def test_mixed_case_toggle_affects_tfl(self):
        txt = "Transport for London (TfL) runs the Tube. TfL operates buses."

        # Mixed-case enabled → should keep the original casing key "TfL"
        mc_on = AcronymDetector(AcronymDetectorConfig(enable_mixed_case=True)).detect(txt)
        on_keys = set(mc_on.unique_acronyms.keys())
        assert "TfL" in on_keys, f"Expected 'TfL' with mixed-case enabled, got {on_keys}"

        # Mixed-case disabled → should not surface TfL (nor an uppercased TFL)
        mc_off = AcronymDetector(AcronymDetectorConfig(enable_mixed_case=False)).detect(txt)
        off_keys = set(mc_off.unique_acronyms.keys())
        assert "TFL" not in off_keys, f"Did not expect 'TFL' with mixed-case disabled, got {off_keys}"
        assert "TfL" not in off_keys, f"Did not expect 'TfL' with mixed-case disabled, got {off_keys}"
