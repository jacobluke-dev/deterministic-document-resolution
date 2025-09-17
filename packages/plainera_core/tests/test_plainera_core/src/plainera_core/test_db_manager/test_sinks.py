import asyncio
from unittest import mock
from unittest.mock import AsyncMock, Mock

import plainera_core.db_manager.sinks as sinks_mod
import pytest
from plainera_core.db_manager.sinks import (
    CompositeSink,
    RouterSink,
    SqlAlchemyModelSink,
    UniversalSink,
    make_sink,
)


# --- fake session
class FakeAsyncSession:
    def __init__(self):
        self.executed = None
        self.committed = False

    async def execute(self, arg):
        self.executed = arg

    async def commit(self):
        self.committed = True


fake_session = FakeAsyncSession()


# --- async CM returned by sessionmaker.begin()
class BeginCM:
    def __init__(self, session):
        self.session = session
        self.entered = 0

    async def __aenter__(self):
        self.entered += 1
        return self.session

    async def __aexit__(self, exc_type, exc, tb):
        if exc_type is None:
            await self.session.commit()
        return False


# --- fake sessionmaker with .begin()
class FakeSessionmaker:
    def __init__(self, cm):
        self._cm = cm
        self.begin_calls = 0

    def begin(self):
        self.begin_calls += 1
        return self._cm


# stub sqlalchemy.insert to record model+row
called = {}


class _InsertStub:
    def __init__(self, model):
        called["model"] = model

    def values(self, **row):
        called["row"] = row
        return ("INSERT", called["model"], row)


@pytest.mark.asyncio
class TestSqlAlchemyModelSink:
    async def test_enqueue_async_maps_executes_and_commits(self, monkeypatch):
        # --- fakes/mocks
        payload = {"event": "hello"}
        mapped = {"k": "v"}

        # mapper should be called with payload and return mapped row
        mapper = mock.Mock(return_value=mapped)

        monkeypatch.setattr("plainera_core.db_manager.sinks.insert", _InsertStub)

        class Model:  # sentinel
            pass

        sessionmaker = FakeSessionmaker(BeginCM(fake_session))
        sink = SqlAlchemyModelSink(sessionmaker, Model, mapper)

        # act
        await sink.enqueue_async(payload)

        # assert
        mapper.assert_called_once_with(payload)
        assert sessionmaker.begin_calls == 1
        assert called["model"] is Model
        assert called["row"] == mapped
        assert fake_session.executed == ("INSERT", Model, mapped)
        assert fake_session.committed is True


class TestUniversalSink:
    @pytest.mark.asyncio
    async def test_enqueue_async_delegates_to_async_sink_only(self):
        payload = {"event": "hello"}

        async_sink = Mock()
        async_sink.enqueue_async = AsyncMock(return_value=None)

        sync_sink = Mock()
        sync_sink.enqueue = Mock()

        sut = UniversalSink(async_sink=async_sink, sync_sink=sync_sink)

        await sut.enqueue_async(payload)

        async_sink.enqueue_async.assert_awaited_once_with(payload)
        sync_sink.enqueue.assert_not_called()

    def test_enqueue_delegates_to_sync_sink_only(self):
        payload = {"event": "hello"}

        async_sink = Mock()
        async_sink.enqueue_async = AsyncMock(return_value=None)

        sync_sink = Mock()
        sync_sink.enqueue = Mock()

        sut = UniversalSink(async_sink=async_sink, sync_sink=sync_sink)

        sut.enqueue(payload)

        sync_sink.enqueue.assert_called_once_with(payload)
        async_sink.enqueue_async.assert_not_called()


class TestMakeSink:
    def test_builds_sink_from_registry_and_passes_through_args(self, monkeypatch):
        # --- Arrange: stub registry with deterministic order
        class ModelA: ...
        class ModelB: ...

        def mapper_a(x): return {"a": 1}
        def mapper_b(x): return {"b": 2}

        monkeypatch.setattr(
            sinks_mod,
            "_SINK_REGISTRY",
            {"a": (ModelA, mapper_a), "b": (ModelB, mapper_b)},
            raising=True,
        )

        # Test double for SqlAlchemyModelSink constructor
        constructed = {}
        class SinkDouble:
            def __init__(self, sessionmaker, model, mapper):
                constructed["sessionmaker"] = sessionmaker
                constructed["model"] = model
                constructed["mapper"] = mapper
            # represent a concrete instance
        sentinel_sessionmaker = object()
        monkeypatch.setattr(sinks_mod, "SqlAlchemyModelSink", SinkDouble, raising=True)

        # --- Act
        sink = make_sink(sentinel_sessionmaker, "b")

        # --- Assert
        assert isinstance(sink, SinkDouble)
        assert constructed["sessionmaker"] is sentinel_sessionmaker
        assert constructed["model"] is ModelB
        assert constructed["mapper"] is mapper_b

    @pytest.mark.parametrize("bad_name", ["", "nope", "logger", "package"])  # names not in our stubbed registry
    def test_raises_value_error_with_valid_names_listed(self, bad_name, monkeypatch):
        class ModelA: ...
        def mapper_a(x): return x

        class ModelZ: ...
        def mapper_z(x): return x

        # Use unsorted keys to verify message sorts them
        monkeypatch.setattr(
            sinks_mod,
            "_SINK_REGISTRY",
            {"zeta": (ModelZ, mapper_z), "alpha": (ModelA, mapper_a)},
            raising=True,
        )

        with pytest.raises(ValueError) as exc:
            make_sink(object(), bad_name)

        msg = str(exc.value)
        # Mentions the bad name and the sorted valid list
        assert "Unknown sink" in msg
        assert "Valid: alpha, zeta" in msg


class TestCompositeSink:
    @pytest.mark.asyncio
    async def test_forwards_to_all_sinks_and_schedules_coroutines(self, monkeypatch):
        # sink A: synchronous enqueue
        sink_a = mock.Mock()
        sink_a.enqueue = mock.Mock(return_value=None)

        # sink B: returns coroutine; should be scheduled via create_task
        async def dummy_coro():
            return None

        coro_instance = dummy_coro()
        sink_b = mock.Mock()
        sink_b.enqueue = mock.Mock(return_value=coro_instance)

        create_task_mock = mock.Mock()
        monkeypatch.setattr(asyncio, "create_task", create_task_mock)

        # SUT – according to docstring, CompositeSink should fan out to multiple sinks.
        # We'll inject sinks the simplest way (as the class under test currently exposes _sinks).
        comp = CompositeSink()
        comp._sinks = [sink_a, sink_b]  # list of sinks we expect to be iterated

        payload = {"event": "x"}
        comp.enqueue(payload)

        # asserts: both sinks received the payload
        sink_a.enqueue.assert_called_once_with(payload)
        sink_b.enqueue.assert_called_once_with(payload)
        # coroutine from sink_b was scheduled
        create_task_mock.assert_called_once_with(coro_instance)


class TestRouterSink:
    def test_routes_payloads_by_predicate_and_schedules_coroutines(self, monkeypatch):
        # predicates
        def only_error(p):
            return p.get("level") == "error"

        def only_info(p):
            return p.get("level") == "info"

        # sinks
        sink_err = mock.Mock()
        sink_inf = mock.Mock()

        # info sink returns a coroutine to ensure scheduling path is exercised
        async def info_coro():
            return None

        info_coro_instance = info_coro()
        sink_inf.enqueue = mock.Mock(return_value=info_coro_instance)
        sink_err.enqueue = mock.Mock(return_value=None)

        create_task_mock = mock.Mock()
        monkeypatch.setattr(asyncio, "create_task", create_task_mock)

        router = RouterSink()
        router._routes = [(only_error, sink_err), (only_info, sink_inf)]

        # 1) info payload
        payload_info = {"level": "info", "event": "startup"}
        router.enqueue(payload_info)
        sink_inf.enqueue.assert_called_once_with(payload_info)
        sink_err.enqueue.assert_not_called()
        create_task_mock.assert_called_once_with(info_coro_instance)

        # 2) error payload
        create_task_mock.reset_mock()
        sink_err.enqueue.reset_mock()
        payload_err = {"level": "error", "event": "crash"}
        router.enqueue(payload_err)
        sink_err.enqueue.assert_called_once_with(payload_err)
        create_task_mock.assert_not_called()
