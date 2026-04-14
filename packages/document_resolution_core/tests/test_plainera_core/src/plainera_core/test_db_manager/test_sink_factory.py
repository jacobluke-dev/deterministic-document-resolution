import types
from collections import defaultdict
from typing import Any, DefaultDict, Optional, TypedDict

import document_resolution_core.db_manager.sink_factory as sink_f
import pytest


# --- dummies -----------------------------------------------------------------
class DummySessionmaker: ...
class DummyModelA: ...
class DummyModelB: ...


# --- shared helpers ----------------------------------------------------------
@pytest.fixture(autouse=True)
def _clean_caches():
    sink_f._MAPPER_CACHE.clear()
    # Do not wipe _REGISTRY globally; tests that need a clean one call set_registry()


Key = tuple[type[Any], str]
CallLog = DefaultDict[Key, list[int]]


def set_registry(monkeypatch, mapping: dict[str, object]) -> None:
    monkeypatch.setattr(sink_f, "_REGISTRY", dict(mapping), raising=True)

def set_cache(monkeypatch, mapping: dict[Key, object]) -> None:
    monkeypatch.setattr(sink_f, "_MAPPER_CACHE", dict(mapping), raising=True)

def expect_unknown_msg(exc: Exception, valid_keys: list[str], unknown: str):
    msg = str(exc)
    assert f"Unknown sink '{unknown}'" in msg
    assert f"Valid: {', '.join(sorted(valid_keys))}" in msg


def _stub_factory(call_log):
    """
    Return a make_logger_mapper stub that records calls and returns a unique
    callable per (model, default).
    """
    def stub(model, *, default_logger_type="decorator"):
        tag = (model, default_logger_type)
        call_log[tag].append(1)

        def mapper(payload):
            return {"_tag": tag, **(payload or {})}

        mapper.__name__ = f"mapper_{id(model)}_{default_logger_type}"
        return mapper
    return stub


# --- _mapper_for --------------------------------------------------------------
class TestMapperFor:
    def test_returns_callable_and_caches_per_key(self, monkeypatch):
        calls: CallLog = defaultdict(list)
        monkeypatch.setattr(sink_f, "make_logger_mapper", _stub_factory(calls))

        m1 = sink_f._mapper_for(DummyModelA, "x")
        assert callable(m1)
        assert calls[(DummyModelA, "x")] == [1]
        assert (DummyModelA, "x") in sink_f._MAPPER_CACHE

        m2 = sink_f._mapper_for(DummyModelA, "x")
        assert m2 is m1
        assert calls[(DummyModelA, "x")] == [1]  # still one factory call

    def test_different_default_logger_type_is_distinct(self, monkeypatch):
        calls: CallLog = defaultdict(list)
        monkeypatch.setattr(sink_f, "make_logger_mapper", _stub_factory(calls))

        mx = sink_f._mapper_for(DummyModelA, "x")
        my = sink_f._mapper_for(DummyModelA, "y")

        assert mx is not my
        assert calls[(DummyModelA, "x")] == [1]
        assert calls[(DummyModelA, "y")] == [1]
        assert (DummyModelA, "x") in sink_f._MAPPER_CACHE
        assert (DummyModelA, "y") in sink_f._MAPPER_CACHE

    def test_different_model_is_distinct(self, monkeypatch):
        calls: CallLog = defaultdict(list)
        monkeypatch.setattr(sink_f, "make_logger_mapper", _stub_factory(calls))

        ax = sink_f._mapper_for(DummyModelA, "x")
        bx = sink_f._mapper_for(DummyModelB, "x")

        assert ax is not bx
        assert calls[(DummyModelA, "x")] == [1]
        assert calls[(DummyModelB, "x")] == [1]
        assert (DummyModelA, "x") in sink_f._MAPPER_CACHE
        assert (DummyModelB, "x") in sink_f._MAPPER_CACHE

    def test_cache_is_used_even_if_factory_changes(self, monkeypatch):
        calls1: CallLog = defaultdict(list)
        monkeypatch.setattr(sink_f, "make_logger_mapper", _stub_factory(calls1))

        first = sink_f._mapper_for(DummyModelA, "x")
        assert calls1[(DummyModelA, "x")] == [1]

        # Swap in a new factory; cached key should NOT invoke it.
        def new_factory(model, *, default_logger_type="decorator"):
            fn = types.SimpleNamespace(tag=("NEW", model, default_logger_type))
            def mapper(payload):
                return {"_new": True, "_tag": fn.tag}
            return mapper

        monkeypatch.setattr(sink_f, "make_logger_mapper", new_factory)

        again = sink_f._mapper_for(DummyModelA, "x")
        assert again is first  # cache hit

        fresh = sink_f._mapper_for(DummyModelA, "y")
        assert fresh is not first


# --- available_sinks ----------------------------------------------------------
class TestAvailableSinks:
    @pytest.mark.parametrize(
        "reg, expected",
        [
            ({"zeta": object(), "alpha": object(), "mid": object()}, ["alpha", "mid", "zeta"]),
            ({"b": object(), "a": object()}, ["a", "b"]),
            ({"logger": object()}, ["logger"]),
        ],
    )
    def test_returns_sorted_keys_and_is_pure(self, monkeypatch, reg, expected):
        set_registry(monkeypatch, reg)
        out1 = sink_f.available_sinks()
        assert out1 == expected

        # mutating the result shouldn't affect subsequent calls
        out1.append("zzz")
        out2 = sink_f.available_sinks()
        assert out2 == expected


# --- make_sink ----------------------------------------------------------------
class TestMakeSink:
    def test_happy_path_constructs_sink_with_model_and_mapper(self, monkeypatch):
        spec = sink_f.SinkSpec(model=DummyModelA, default_logger_type="alpha")
        set_registry(monkeypatch, {"foo": spec})

        def mapper_sentinel(payload: CallLog) -> dict[str, bool]:
            return {"ok": True}

        called = {}
        def _stub_mapper_for(model, default_logger_type):
            called["args"] = (model, default_logger_type)
            return mapper_sentinel
        monkeypatch.setattr(sink_f, "_mapper_for", _stub_mapper_for, raising=True)

        captured = {}
        class StubSink:
            def __init__(self, sessionmaker, model, mapper):
                captured["args"] = (sessionmaker, model, mapper)
        monkeypatch.setattr(sink_f, "SqlAlchemyModelSink", StubSink, raising=True)

        sm = DummySessionmaker()
        sink = sink_f.make_sink(sm, "foo")

        assert called["args"] == (DummyModelA, "alpha")
        assert isinstance(sink, StubSink)
        assert captured["args"] == (sm, DummyModelA, mapper_sentinel)

    def test_unknown_sink_raises_value_error_and_lists_valid(self, monkeypatch):
        reg = {
            "logger": sink_f.SinkSpec(model=DummyModelA, default_logger_type="api"),
            "package_logger": sink_f.SinkSpec(model=DummyModelB, default_logger_type="package"),
        }
        set_registry(monkeypatch, reg)

        with pytest.raises(ValueError) as ei:
            sink_f.make_sink(DummySessionmaker(), "nope")
        expect_unknown_msg(ei.value, list(reg.keys()), "nope")


class _CallState(TypedDict):
    count: int
    args: Optional[tuple[type[Any], str]]

# --- make_universal_sink ------------------------------------------------------
class TestMakeUniversalSink:
    def test_happy_path_builds_async_and_sync_sinks_and_universal(self, monkeypatch):
        spec = sink_f.SinkSpec(model=DummyModelA, default_logger_type="alpha")
        set_registry(monkeypatch, {"foo": spec})

        calls: _CallState = {"count": 0, "args": None}
        mapper_sentinel = object()

        def stub_mapper_for(model, default_logger_type):
            calls["count"] += 1
            calls["args"] = (model, default_logger_type)  # ← no trailing comma
            return mapper_sentinel

        monkeypatch.setattr(sink_f, "_mapper_for", stub_mapper_for, raising=True)

        captured_async = {}
        class StubAsyncSink:
            def __init__(self, sessionmaker, model, mapper):
                captured_async["args"] = (sessionmaker, model, mapper)
        captured_sync = {}
        class StubSyncSink:
            def __init__(self, url, model, mapper):
                captured_sync["args"] = (url, model, mapper)
        monkeypatch.setattr(sink_f, "SqlAlchemyModelSink", StubAsyncSink, raising=True)
        monkeypatch.setattr(sink_f, "SyncSqlAlchemyModelSink", StubSyncSink, raising=True)

        wrapper_captured = {}
        class StubUniversal:
            def __init__(self, async_sink, sync_sink):
                wrapper_captured["args"] = (async_sink, sync_sink)
                self.async_sink = async_sink
                self.sync_sink = sync_sink
        monkeypatch.setattr(sink_f, "UniversalSink", StubUniversal, raising=True)

        sm = DummySessionmaker()
        url = "postgresql+psycopg://db"
        uni = sink_f.make_universal_sink(sm, url, "foo")

        assert calls["count"] == 1
        assert calls["args"] == (DummyModelA, "alpha")
        assert captured_async["args"] == (sm, DummyModelA, mapper_sentinel)
        assert captured_sync["args"] == (url, DummyModelA, mapper_sentinel)
        a, s = wrapper_captured["args"]
        assert uni.async_sink is a
        assert uni.sync_sink is s

    def test_unknown_sink_raises_and_lists_valid_names(self, monkeypatch):
        reg = {
            "logger": sink_f.SinkSpec(model=DummyModelA, default_logger_type="api"),
            "package_logger": sink_f.SinkSpec(model=DummyModelA, default_logger_type="package"),
        }
        set_registry(monkeypatch, reg)

        with pytest.raises(ValueError) as ei:
            sink_f.make_universal_sink(DummySessionmaker(), "postgres://x", "nope")
        expect_unknown_msg(ei.value, list(reg.keys()), "nope")


# --- register_sink ------------------------------------------------------------
class TestRegisterSink:
    def test_adds_entry_and_invalidates_matching_cache(self, monkeypatch):
        set_registry(monkeypatch, {})
        set_cache(monkeypatch, {(DummyModelA, "decorator"): object(), (DummyModelA, "other"): object()})

        sink_f.register_sink("foo", DummyModelA, default_logger_type="decorator")

        assert sink_f._REGISTRY["foo"] == sink_f.SinkSpec(DummyModelA, "decorator")
        assert (DummyModelA, "decorator") not in sink_f._MAPPER_CACHE
        assert (DummyModelA, "other") in sink_f._MAPPER_CACHE

    def test_overwrite_updates_registry_and_invalidates_new_tuple_only(self, monkeypatch):
        set_registry(monkeypatch, {"foo": sink_f.SinkSpec(DummyModelA, "old")})
        old_entry = object()
        new_entry = object()
        set_cache(monkeypatch, {(DummyModelA, "old"): old_entry, (DummyModelB, "new"): new_entry})

        sink_f.register_sink("foo", DummyModelB, default_logger_type="new")

        assert sink_f._REGISTRY["foo"] == sink_f.SinkSpec(DummyModelB, "new")
        assert (DummyModelB, "new") not in sink_f._MAPPER_CACHE
        assert sink_f._MAPPER_CACHE[(DummyModelA, "old")] is old_entry

    def test_no_cache_entry_is_safe_noop_invalidation(self, monkeypatch):
        set_registry(monkeypatch, {})
        set_cache(monkeypatch, {})

        sink_f.register_sink("bar", DummyModelA, default_logger_type="x")

        assert sink_f._REGISTRY["bar"] == sink_f.SinkSpec(DummyModelA, "x")
        assert sink_f._MAPPER_CACHE == {}
