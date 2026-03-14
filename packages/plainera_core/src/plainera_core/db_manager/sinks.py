from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Iterable, Protocol

from sqlalchemy import create_engine, insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker

from plainera_core.db_manager.config import MapperFn

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


# ---- Fan-out and routing ----------------------------------------------------


class CompositeSink:
    """Fan-out sink that forwards log payloads to multiple downstream sinks.

    This sink is useful when we want each log record to be persisted or
    transmitted to more than one backend. For example, we6 might want to
    write the same structured log payload to both a SQL database and
    standard output.

    Each downstream sink is invoked with the same payload. If a sink's
    ``enqueue`` method returns a coroutine, it is scheduled via
    ``asyncio.create_task`` in a fire-and-forget manner, so failures or
    shutdown timing can cause payload loss in short-lived processes.

    Example:
        >>> sink = CompositeSink([SqlAlchemyModelSink(...), ConsoleSink()])
        >>> sink.enqueue({"event": "user_signup", "level": "info"})

    Args:
        sinks (Iterable[DbSink]): A sequence of sink instances that implement
            an ``enqueue`` method accepting a log payload.
    """

    def __init__(self, sinks: Iterable[DbSink] = ()):
        self._sinks: list[DbSink] = list(sinks)

    def enqueue(self, payload: dict[str, Any]) -> None:
        """Forward a log payload to all configured sinks.

        Args:
            payload (dict[str, Any]): The structured log payload to send.

        Notes:
            - If a downstream sink's ``enqueue`` returns a coroutine,
              this method schedules it on the event loop using
              ``asyncio.create_task``.
            - Fire-and-forget means the caller does not wait for all
              downstream sinks to finish writing.
        """
        for s in self._sinks:
            rv = s.enqueue(payload)
            if asyncio.iscoroutine(rv):
                asyncio.create_task(rv)

Predicate = Callable[[dict[str, Any]], bool]

class RouterSink:
    """Routing sink that forwards log payloads only to selected sinks.

    Each sink is associated with a predicate function. When a payload is
    enqueued, the predicate is evaluated; if it returns ``True``, the
    payload is forwarded to that sink. This allows fine-grained routing
    based on log content (e.g., error-level events to the DB, access logs
    to a file).

    Example:
        >>> error_only = lambda p: p.get("level") == "error"
        >>> info_only = lambda p: p.get("level") == "info"
        >>> sink = RouterSink([
        ...     (error_only, SqlAlchemyModelSink(...)),
        ...     (info_only, ),
        ... ])
        >>> sink.enqueue({"event": "startup", "level": "info"})

    Args:
        _routes (list[tuple[Predicate, DbSink]]): A list of (predicate, sink)
            pairs. ``predicate`` is a callable that takes a payload dict
            and returns a boolean. ``sink`` is any sink with an ``enqueue``
            method.
    """

    def __init__(self, routes: Iterable[tuple[Predicate, DbSink]] = ()):
        self._routes: list[tuple[Predicate, DbSink]] = list(routes)

    def enqueue(self, payload: dict[str, Any]) -> None:
        """Forward a log payload to sinks whose predicates match.

        Args:
            payload (dict[str, Any]): The structured log payload to evaluate
                against routing predicates.

        Notes:
            - Multiple predicates can match, in which case the payload is
              sent to multiple sinks.
            - If a downstream sink's ``enqueue`` returns a coroutine,
              it is scheduled with ``asyncio.create_task`` (fire-and-forget).
        """
        for pred, sink in self._routes:
            if pred(payload):
                rv = sink.enqueue(payload)
                if asyncio.iscoroutine(rv):
                    asyncio.create_task(rv)
