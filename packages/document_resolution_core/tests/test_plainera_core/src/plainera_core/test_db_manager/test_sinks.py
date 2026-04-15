from unittest import mock
from unittest.mock import AsyncMock, Mock

import pytest
from document_resolution_core.db_manager.sinks import (
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

        monkeypatch.setattr("document_resolution_core.db_manager.sinks.insert", _InsertStub)

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
