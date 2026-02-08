from types import SimpleNamespace

from plainera_unacronym.nlp.extraction.engine.stages import (
    Chain,
    Stage,
    StageResult,
    Tracer,
)


class DummyState:
    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


class TestChainUnit:
    def test_run_executes_stages_in_order_and_returns_reports(self):
        state = DummyState(x=0)

        def s1(s):
            s.x += 1
            return StageResult(s, note="s1")

        def s2(s):
            s.x *= 10
            return StageResult(s, note="s2")

        chain = Chain(
            [
                Stage(name="one", fn=s1),
                Stage(name="two", fn=s2),
            ]
        )

        out, reports = chain.run(state)

        assert out is state
        assert state.x == 10  # (0 + 1) * 10
        assert [r.name for r in reports] == ["one", "two"]
        assert [r.info for r in reports] == ["s1", "s2"]
        assert all(r.ok for r in reports)

    def test_run_preserves_preview_per_stage(self):
        state = DummyState(x=1)

        def fn(s):
            s.x += 1
            return StageResult(s, note="inc")

        def preview(s):
            return f"x={s.x}"

        chain = Chain([Stage(name="inc", fn=fn, preview=preview)])

        _, reports = chain.run(state)

        assert reports[0].preview == "x=2"


class TestChainTracerIntegration:
    def test_chain_run_records_tracer_events_only_for_stages_that_change_tracked_fields(self):
        # Track picks across multiple stages; only one stage mutates it.
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

        def noop(s):
            return StageResult(s, note="noop")

        def mutate_picks(s):
            s.picks["ema"] = SimpleNamespace(
                definition="European Medicines Agency (v2)",
                original_definition="European Medicines Agency (v2)",
                acr_span=(10, 13),
                def_span=(0, 28),
                confidence=0.80,
            )
            return StageResult(s, note="mutated")

        chain = Chain(
            [
                Stage(name="a_noop", fn=noop, trace_fields=("picks",)),
                Stage(name="b_mutate", fn=mutate_picks, trace_fields=("picks",)),
                Stage(name="c_noop", fn=noop, trace_fields=("picks",)),
            ]
        )

        tracer = Tracer()
        _, reports = chain.run(state, tracer=tracer)

        # Reports are returned for all stages
        assert [r.name for r in reports] == ["a_noop", "b_mutate", "c_noop"]
        assert [r.info for r in reports] == ["noop", "mutated", "noop"]

        # Tracer records only when diff exists: should be exactly one event
        assert len(tracer.events) == 1
        ev = tracer.events[0]
        assert ev.stage == "b_mutate"

        before_rows = ev.before["picks"]
        after_rows = ev.after["picks"]

        assert before_rows[0]["acr"] == "ema"
        assert after_rows[0]["acr"] == "ema"
        assert before_rows[0]["definition"] == "European Medicines Agency"
        assert after_rows[0]["definition"] == "European Medicines Agency (v2)"

    def test_chain_run_allows_multiple_fields_tracing(self):
        # Demonstrates tracer diffing across more than one tracked field.
        state = DummyState(
            picks={"nlp": SimpleNamespace(definition="Natural language processing")},
            anchored_defs=[],
        )

        def stage_add_def(s):
            s.anchored_defs.append(
                SimpleNamespace(
                    acronym="NLP",
                    definition="natural language processing",
                    original_definition="Natural language processing",
                    acr_start=10,
                    acr_end=13,
                    def_start=0,
                    def_end=27,
                    confidence=0.95,
                    source="in_text",
                )
            )
            return StageResult(s, note="added")

        chain = Chain(
            [
                Stage(
                    name="add_def",
                    fn=stage_add_def,
                    trace_fields=("picks", "anchored_defs"),
                )
            ]
        )

        tracer = Tracer()
        chain.run(state, tracer=tracer)

        assert len(tracer.events) == 1
        ev = tracer.events[0]
        assert ev.stage == "add_def"

        assert "picks" in ev.before and "picks" in ev.after
        assert "anchored_defs" in ev.before and "anchored_defs" in ev.after
        assert ev.before["anchored_defs"] == []
        assert len(ev.after["anchored_defs"]) == 1
        assert ev.after["anchored_defs"][0]["acr"] == "NLP"
