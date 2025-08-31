# sinks.py
from typing import Any, Awaitable, Protocol

from db.models import Logger
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


class DbSink(Protocol):
    def enqueue(self, payload: dict[str, Any]) -> None | Awaitable[None]: ...


class PostgresDbSink:
    """
    Async DB sink used by observability.emit().
    """
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    def enqueue(self, payload: dict[str, Any]) -> Awaitable[None]:
        return self._write(payload)

    async def _write(self, payload: dict[str, Any]) -> None:
        known = {
            "timestamp","level","event","logger_type","request_id",
            "function","args","duration_ms","result","error",
        }
        row = {k: payload.get(k) for k in known}
        row["extra"] = {k: v for k, v in payload.items() if k not in known}

        async with self._sf() as s:
            await s.execute(insert(Logger).values(**row))
            await s.commit()
