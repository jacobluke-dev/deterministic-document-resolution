import types
from unittest import mock
from unittest.mock import AsyncMock, Mock

import plainera_core.db_manager.sinks as sinks_mod
import pytest
from plainera_core.db_manager.sinks import (
    CompositeSink,
    RouterSink,
    SqlAlchemyModelSink,
    UniversalSink,
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


class TestCompositeSink:
    @pytest.mark.asyncio
    async def test_forwards_to_all_sinks_and_schedules_coroutines(self, monkeypatch):
        # sink A: synchronous
        sink_a = mock.Mock()
        sink_a.enqueue = mock.Mock(return_value=None)

        # sink B: returns a FRESH coroutine each call
        async def dummy_coro():
            return None

        sink_b = mock.Mock()
        sink_b.enqueue = mock.Mock(side_effect=lambda payload: dummy_coro())

        # Patch the symbol actually used by CompositeSink
        # (handles either `import asyncio` or `from asyncio import create_task`)
        def _consume(coro):
            # close the coroutine so we don't get "was never awaited"
            coro.close()
            return object()  # fake task/sentinel

        create_task_mock = mock.Mock(side_effect=_consume)
        if hasattr(sinks_mod, "create_task"):
            monkeypatch.setattr(sinks_mod, "create_task", create_task_mock)
        else:
            monkeypatch.setattr(sinks_mod.asyncio, "create_task", create_task_mock)

        # SUT
        comp = CompositeSink()
        comp._sinks = [sink_a, sink_b]

        payload = {"event": "x"}
        comp.enqueue(payload)

        # both sinks receive payload
        sink_a.enqueue.assert_called_once_with(payload)
        sink_b.enqueue.assert_called_once_with(payload)

        # scheduled a coroutine
        (arg,), _ = create_task_mock.call_args
        assert isinstance(arg, types.CoroutineType)


class TestRouterSink:

    def test_routes_payloads_by_predicate_and_schedules_coroutines(self, monkeypatch):
        # predicates
        def only_error(p): return p.get("level") == "error"
        def only_info(p):  return p.get("level") == "info"

        # sinks
        sink_err = mock.Mock()
        sink_inf = mock.Mock()

        # info sink returns a coroutine to ensure scheduling path is exercised
        async def info_coro():
            return None

        # return a FRESH coroutine each time (no stored instance)
        sink_inf.enqueue = mock.Mock(side_effect=lambda payload: info_coro())
        sink_err.enqueue = mock.Mock(return_value=None)

        # Patch the EXACT symbol RouterSink uses:
        # - If the module did `import asyncio`, patch sinks_mod.asyncio.create_task
        # - If it did `from asyncio import create_task`, patch sinks_mod.create_task
        create_task_mock = mock.Mock(side_effect=lambda c: c.close())  # consume coro to avoid warnings
        if hasattr(sinks_mod, "create_task"):
            monkeypatch.setattr(sinks_mod, "create_task", create_task_mock)
        else:
            monkeypatch.setattr(sinks_mod.asyncio, "create_task", create_task_mock)

        router = RouterSink()
        router._routes = [(only_error, sink_err), (only_info, sink_inf)]

        # 1) info payload -> schedules coroutine
        payload_info = {"level": "info", "event": "startup"}
        router.enqueue(payload_info)
        sink_inf.enqueue.assert_called_once_with(payload_info)
        sink_err.enqueue.assert_not_called()
        # assert it was passed *a coroutine*
        passed = create_task_mock.call_args.args[0]
        assert isinstance(passed, types.CoroutineType)

        # 2) error payload -> no scheduling
        create_task_mock.reset_mock()
        sink_err.enqueue.reset_mock()
        payload_err = {"level": "error", "event": "crash"}
        router.enqueue(payload_err)
        sink_err.enqueue.assert_called_once_with(payload_err)
        create_task_mock.assert_not_called()
