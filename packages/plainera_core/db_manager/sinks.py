import asyncio
from typing import Any, Awaitable, Callable, Iterable, Protocol

from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

# ---- Protocol ---------------------------------------------------------------

class DbSink(Protocol):
    def enqueue(self, payload: dict[str, Any]) -> None | Awaitable[None]: ...

Predicate = Callable[[dict[str, Any]], bool]
MapPayload = Callable[[dict[str, Any]], dict[str, Any]]

# ---- Single-model sink (async SQLAlchemy) -----------------------------------

class SqlAlchemyModelSink:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        model_class: type[Any],               # explicit
        map_payload: MapPayload,
    ) -> None:
        self._sf = session_factory
        self._model = model_class
        self._map = map_payload

    def enqueue(self, payload: dict[str, Any]) -> Awaitable[None]:
        return self._write(payload)

    async def _write(self, payload: dict[str, Any]) -> None:
        row = self._map(payload)
        async with self._sf() as s:
            await s.execute(insert(self._model).values(**row))
            await s.commit()

# ---- Fan-out and routing ----------------------------------------------------

class CompositeSink:
    """
    Send every payload to all sinks (fan-out).
    """
    def __init__(self, sinks: Iterable[DbSink]) -> None:
        self._sinks = list(sinks)

    def enqueue(self, payload: dict[str, Any]) -> None:
        for s in self._sinks:
            rv = s.enqueue(payload)
            if asyncio.iscoroutine(rv):
                asyncio.create_task(rv)

class RouterSink:
    """
    Route payloads to selected sinks based on a predicate.
    """
    def __init__(self, routes: list[tuple[Predicate, DbSink]]) -> None:
        self._routes = routes

    def enqueue(self, payload: dict[str, Any]) -> None:
        for pred, sink in self._routes:
            if pred(payload):
                rv = sink.enqueue(payload)
                if asyncio.iscoroutine(rv):
                    asyncio.create_task(rv)
