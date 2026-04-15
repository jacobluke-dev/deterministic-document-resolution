from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, Protocol

from sqlalchemy import create_engine, insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker

from document_resolution_core.db_manager.config import MapperFn

# ---- Protocol ---------------------------------------------------------------

class DbSink(Protocol):
    def enqueue(self, payload: dict[str, Any]) -> None | Awaitable[None]: ...


class SqlAlchemyModelSink:
    """
    Async sink (use inside async code; awaited by emit_async / decorator).
    """

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession], model: type[Any], mapper: MapperFn):
        self._Session = sessionmaker
        self._model = model
        self._map = mapper

    async def enqueue_async(self, payload: dict[str, Any]) -> None:
        row = self._map(payload)
        async with self._Session.begin() as s:  # creates session + transaction
            await s.execute(insert(self._model).values(**row))


class SyncSqlAlchemyModelSink:
    """
    Sync sink (call from plain sync code; no event loop needed).
    """

    def __init__(self, url: str, model: type[Any], mapper: MapperFn):
        # Use psycopg (sync) URL, e.g. postgresql+psycopg://...
        self._engine = create_engine(url, pool_pre_ping=True, future=True)
        self._Session = sessionmaker(self._engine, expire_on_commit=False)
        self._model = model
        self._map = mapper

    def enqueue(self, payload: dict[str, Any]) -> None:
        row = self._map(payload)
        with self._Session.begin() as s:
            s.add(self._model(**row))


class UniversalSink:
    """Exposes both .enqueue_async() and .enqueue() by delegating to the right
    backend.

    Safe to pass to BOTH the async decorator and message_logger.
    """

    def __init__(self, async_sink: SqlAlchemyModelSink, sync_sink: SyncSqlAlchemyModelSink):
        self._async = async_sink
        self._sync = sync_sink

    async def enqueue_async(self, payload: dict[str, Any]) -> None:
        await self._async.enqueue_async(payload)

    def enqueue(self, payload: dict[str, Any]) -> None:
        self._sync.enqueue(payload)
