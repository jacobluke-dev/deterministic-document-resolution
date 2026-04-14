from types import SimpleNamespace

import pytest
from plainera_unacronym.nlp.extraction.base.stages import TraceEvent, Tracer


class DummyState:
    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


class TestTracerUnit:
    def test_snapshot_includes_only_existing_fields(self):
        state = DummyState(picks={"ema": SimpleNamespace(definition="European Medicines Agency")})
        t = Tracer()

        snap = t.snapshot(state, fields=["picks", "anchored_defs", "missing_field"])

        assert "picks" in snap
        assert "anchored_defs" not in snap  # not present on state
        assert "missing_field" not in snap  # not present on state

    def test_record_does_nothing_when_before_or_after_none(self):
        t = Tracer()
        t.record("stage_x", None, {"a": 1})
        t.record("stage_x", {"a": 1}, None)
        assert t.events == []

    def test_record_does_nothing_when_snapshots_equal(self):
        t = Tracer()
        before = {"picks": [{"acr": "ema"}]}
        after = {"picks": [{"acr": "ema"}]}

        t.record("stage_x", before, after)
        assert t.events == []

    def test_record_appends_event_when_snapshots_differ(self):
        t = Tracer()
        before = {"picks": [{"acr": "ema", "definition": "A"}]}
        after = {"picks": [{"acr": "ema", "definition": "B"}]}

        t.record("stage_x", before, after)
        assert len(t.events) == 1
        ev = t.events[0]
        assert isinstance(ev, TraceEvent)
        assert ev.stage == "stage_x"
        assert ev.before == before
        assert ev.after == after

    def test_snap_value_dict_omits_none_picks(self):
        picks = {
            "ema": None,
            "nlp": SimpleNamespace(
                definition="Natural language processing",
                original_definition="Natural language processing",
                acr_span=(10, 13),
                def_span=(0, 27),
                confidence=0.95,
            ),
        }
        t = Tracer()
        out = t._snap_value(picks)

        assert isinstance(out, list)
        assert len(out) == 1
        row = out[0]
        assert row["acr"] == "nlp"  # NOTE: dict key, not pick.acronym
        assert row["definition"] == "Natural language processing"
        assert row["orig"] == "Natural language processing"
        assert row["acr_span"] == (10, 13)
        assert row["def_span"] == (0, 27)
        assert row["conf"] == pytest.approx(0.95)

    def test_snap_value_list_maps_def_rows(self):
        defs = [
            SimpleNamespace(
                acronym="EMA",
                definition="european medicines agency",
                original_definition="European Medicines Agency",
                acr_start=30,
                acr_end=33,
                def_start=0,
                def_end=24,
                confidence=0.95,
                source="all_occ_scan_parenthetical",
            )
        ]
        t = Tracer()
        out = t._snap_value(defs)

        assert isinstance(out, list)
        assert len(out) == 1
        row = out[0]
        assert row["acr"] == "EMA"
        assert row["definition"] == "european medicines agency"
        assert row["orig"] == "European Medicines Agency"
        assert row["spans"] == (30, 33, 0, 24)
        assert row["conf"] == pytest.approx(0.95)
        assert row["src"] == "all_occ_scan_parenthetical"

    def test_filter_regex_applies_to_dict_keys_for_picks(self):
        picks = {
            "ema": SimpleNamespace(definition="European Medicines Agency"),
            "nlp": SimpleNamespace(definition="Natural language processing"),
        }
        t = Tracer(filter_regex=r"^ema$")
        out = t._snap_value(picks)

        assert [r["acr"] for r in out] == ["ema"]

    def test_filter_regex_applies_to_item_acronym_for_defs(self):
        defs = [
            SimpleNamespace(
                acronym="EMA",
                definition="x",
                original_definition="x",
                acr_start=0,
                acr_end=1,
                def_start=0,
                def_end=1,
                confidence=0.9,
                source="all_occ_scan_parenthetical",
            ),
            SimpleNamespace(
                acronym="NLP",
                definition="y",
                original_definition="y",
                acr_start=0,
                acr_end=1,
                def_start=0,
                def_end=1,
                confidence=0.9,
                source="all_occ_scan_parenthetical",
            ),
        ]
        t = Tracer(filter_regex=r"^EMA$")
        out = t._snap_value(defs)

        assert len(out) == 1
        assert out[0]["acr"] == "EMA"

    def test_filter_regex_skips_defs_with_no_acronym_attr_when_enabled(self):
        defs = [
            SimpleNamespace(definition="x"),  # no acronym attribute
            SimpleNamespace(acronym="EMA", definition="y"),
        ]
        t = Tracer(filter_regex=r"EMA")
        out = t._snap_value(defs)

        assert len(out) == 1
        assert out[0]["acr"] == "EMA"

    def test_snap_value_returns_none_for_unsupported_types(self):
        t = Tracer()
        assert t._snap_value("not-supported") is None
        assert t._snap_value(123) is None
        assert t._snap_value(object()) is None

    def test_snapshot_normalises_values_via_snap_value(self):
        picks = {"ema": SimpleNamespace(definition="European Medicines Agency")}
        defs = [
            SimpleNamespace(
                acronym="EMA",
                definition="x",
                original_definition="x",
                acr_start=0,
                acr_end=1,
                def_start=0,
                def_end=1,
                confidence=0.9,
                source="all_occ_scan_parenthetical",
            )
        ]
        state = DummyState(picks=picks, anchored_defs=defs)

        t = Tracer()
        snap = t.snapshot(state, fields=["picks", "anchored_defs"])

        assert isinstance(snap["picks"], list)
        assert isinstance(snap["anchored_defs"], list)
        assert snap["picks"][0]["acr"] == "ema"
        assert snap["anchored_defs"][0]["acr"] == "EMA"
