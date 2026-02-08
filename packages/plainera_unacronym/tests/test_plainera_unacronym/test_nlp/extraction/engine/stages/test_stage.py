
from types import SimpleNamespace

from plainera_unacronym.nlp.extraction.engine.stages import Stage, StageReport, StageResult, Tracer


class DummyState:
    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


class SpyTracer:
    """Minimal tracer spy capturing snapshot/record calls."""

    def __init__(self, snapshots=None):
        # Optional fixed snapshots to return sequentially.
        self._snapshots = list(snapshots) if snapshots is not None else None
        self.snapshot_calls = []
        self.record_calls = []

    def snapshot(self, state, fields):
        self.snapshot_calls.append((state, tuple(fields)))
        if self._snapshots is None:
            return {}
        return self._snapshots.pop(0)

    def record(self, stage, before, after):
        self.record_calls.append((stage, before, after))


class TestStageUnit:
    def test_run_calls_fn_and_returns_state_and_report(self):
        state = DummyState(x=1)

        def fn(s):
            s.x += 1
            return StageResult(s, note="bumped")

        st = Stage(name="detect", fn=fn)

        out_state, report = st.run(state)

        assert out_state is state
        assert state.x == 2
        assert isinstance(report, StageReport)
        assert report.name == "detect"
        assert report.ok is True
        assert report.info == "bumped"
        assert report.preview is None

    def test_run_sets_preview_when_preview_fn_present(self):
        state = DummyState(count=0)

        def fn(s):
            s.count = 5
            return StageResult(s, note="ok")

        def preview(s):
            return f"count={s.count}"

        st = Stage(name="merge", fn=fn, preview=preview)

        _, report = st.run(state)

        assert report.preview == "count=5"

    def test_run_with_tracer_snapshots_before_and_after_and_records(self):
        state = DummyState(picks={"ema": 1})

        def fn(s):
            s.picks["ema"] = 2
            return StageResult(s, note="updated")

        st = Stage(name="picks", fn=fn, trace_fields=("picks",))

        tracer = SpyTracer(snapshots=[{"picks": [1]}, {"picks": [2]}])

        out_state, report = st.run(state, tracer=tracer)

        assert out_state is state
        assert report.name == "picks"

        # snapshot called twice: before + after
        assert len(tracer.snapshot_calls) == 2
        (s0, fields0) = tracer.snapshot_calls[0]
        (s1, fields1) = tracer.snapshot_calls[1]
        assert s0 is state
        assert s1 is state  # Stage mutates in-place; result.value is same object
        assert fields0 == ("picks",)
        assert fields1 == ("picks",)

        # record called once with stage name and returned snapshots
        assert tracer.record_calls == [("picks", {"picks": [1]}, {"picks": [2]})]

    def test_run_without_tracer_does_not_snapshot_or_record(self):
        state = DummyState(x=0)

        def fn(s):
            s.x = 1
            return StageResult(s, note="ok")

        st = Stage(name="detect", fn=fn, trace_fields=("x",))

        # If tracer isn't passed, no tracer should be invoked at all.
        out_state, report = st.run(state, tracer=None)

        assert out_state is state
        assert report.name == "detect"
        assert state.x == 1

    def test_run_tracer_with_empty_trace_fields_still_calls_snapshot_and_record(self):
        state = DummyState(x=0)

        def fn(s):
            s.x = 1
            return StageResult(s, note="ok")

        st = Stage(name="stage_x", fn=fn, trace_fields=())

        tracer = SpyTracer(snapshots=[{}, {}])

        st.run(state, tracer=tracer)

        # Snapshot called even if trace_fields empty (because run() calls snapshot unconditionally if tracer)
        assert tracer.snapshot_calls == [(state, ()), (state, ())]
        # record called with the empty snapshots; real Tracer would likely suppress if equal
        assert tracer.record_calls == [("stage_x", {}, {})]

    def test_run_passes_post_stage_state_to_preview(self):
        state = DummyState(v=10)

        def fn(s):
            s.v = 42
            return StageResult(s, note="done")

        seen = {}

        def preview(s):
            seen["v"] = s.v
            return "pv"

        st = Stage(name="pv_stage", fn=fn, preview=preview)

        st.run(state)

        assert seen["v"] == 42


class TestStageTracerIntegration:
    def test_stage_run_records_tracer_event_only_when_tracked_field_changes(self):
        # picks is the structure Tracer knows how to normalise (dict -> list of row dicts)
        state = DummyState(
            picks={
                "ema": SimpleNamespace(
                    definition="European Medicines Agency",
                    original_definition="European Medicines Agency",
                    acr_span=(10, 13),
                    def_span=(0, 24),
                    confidence=0.95,
                )
            }
        )

        def fn_no_change(s):
            # No mutation => tracer should see identical snapshots => no events
            return StageResult(s, note="noop")

        def fn_change(s):
            # Mutate the tracked field => tracer should record exactly one event
            s.picks["ema"] = SimpleNamespace(
                definition="European Medicines Agency (v2)",
                original_definition="European Medicines Agency (v2)",
                acr_span=(10, 13),
                def_span=(0, 28),
                confidence=0.80,
            )
            return StageResult(s, note="changed")

        stage = Stage(name="picks_stage", fn=fn_no_change, trace_fields=("picks",))
        tracer = Tracer()

        # 1) No change => no events
        _, report1 = stage.run(state, tracer=tracer)
        assert report1.ok is True
        assert report1.name == "picks_stage"
        assert len(tracer.events) == 0

        # 2) Change => one event
        stage.fn = fn_change
        _, report2 = stage.run(state, tracer=tracer)
        assert report2.ok is True
        assert report2.info == "changed"
        assert len(tracer.events) == 1

        ev = tracer.events[0]
        assert ev.stage == "picks_stage"

        # Tracer normalises picks dict into list of row dicts.
        # The dict key becomes "acr" in the row (important: key, not pick.acronym)
        before_rows = ev.before["picks"]
        after_rows = ev.after["picks"]

        assert len(before_rows) == 1
        assert len(after_rows) == 1
        assert before_rows[0]["acr"] == "ema"
        assert after_rows[0]["acr"] == "ema"
        assert before_rows[0]["definition"] == "European Medicines Agency"
        assert after_rows[0]["definition"] == "European Medicines Agency (v2)"
