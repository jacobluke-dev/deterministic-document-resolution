import asyncio
from unittest import mock

import pytest

# Adjust import path if needed
from plainera_core.db_manager.sinks import (
    CompositeSink,
    RouterSink,
    SqlAlchemyModelSink,
)


@pytest.mark.unit
class TestSqlAlchemyModelSink:
    @pytest.mark.asyncio
    async def test_enqueue_async_maps_executes_and_commits(self, monkeypatch):
        # --- fakes/mocks
        payload = {"event": "hello"}
        mapped = {"k": "v"}

        # mapper should be called with payload and return mapped row
        mapper = mock.Mock(return_value=mapped)

        # minimal async session fake
        class FakeAsyncSession:
            def __init__(self):
                self.executed = None
                self.committed = False

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def execute(self, arg):
                self.executed = arg

            async def commit(self):
                self.committed = True

        fake_session = FakeAsyncSession()

        # session_factory returns an async context manager (our FakeAsyncSession)
        session_factory = mock.Mock(return_value=fake_session)

        # intercept sqlalchemy.insert(...) so we can see model passed
        called = {}

        class _InsertStub:
            def __init__(self, model):
                called["model"] = model

            def values(self, **row):
                called["row"] = row
                return ("INSERT", called["model"], row)

        def insert_stub(model):
            return _InsertStub(model)

        monkeypatch.setattr("plainera_core.db_manager.sinks.insert", insert_stub)

        # SUT
        class Model:  # sentinel for model class
            pass

        sink = SqlAlchemyModelSink(session_factory, Model, mapper)

        # act
        await sink.enqueue_async(payload)

        # assert
        mapper.assert_called_once_with(payload)
        # session_factory was used
        session_factory.assert_called_once()
        # our stub recorded model + row
        assert called["model"] is Model
        assert called["row"] == mapped
        # session methods invoked
        assert fake_session.executed == ("INSERT", Model, mapped)
        assert fake_session.committed is True


@pytest.mark.unit
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


@pytest.mark.unit
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
