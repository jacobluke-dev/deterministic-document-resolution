"""Structured log emission utilities.

Builds a JSON payload with a UTC timestamp, normalized level, request id,
and scrubbed fields; writes it to the standard logger and (optionally)
forwards it to a database-backed sink. Errors in sink submission are
swallowed to avoid impacting application flow.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from db_manager.sinks import DbSink, AsyncDbSink
from .levels import LogLevel, STD_LEVEL
from .context import request_id_var
from .redact import scrub

logger = logging.getLogger("plainera")


# emit.py
def _payload(event: str, level: LogLevel | str, logger_type: str, fields: dict[str, Any]) -> dict[str, Any]:
    lvl = level.name.lower() if isinstance(level, LogLevel) else str(level).lower()
    ts = datetime.now(timezone.utc).isoformat()
    data = {
        "timestamp": ts,
        "level": lvl,
        "event": event,
        "logger_type": logger_type,
        "request_id": request_id_var.get(),
    }
    if fields:
        data.update(scrub(fields))
    return data


def _log_std(level: LogLevel | str, payload: dict[str, Any]) -> None:
    # Map to std level only if needed
    std_level = STD_LEVEL[level] if isinstance(level, LogLevel) else STD_LEVEL[LogLevel[str(level).upper()]]
    if logger.isEnabledFor(std_level):
        logger.log(std_level, json.dumps(payload, default=str))


def emit(event: str, *, level: LogLevel = LogLevel.INFO, logger_type: str = "decorator",
         db_sink: Optional[AsyncDbSink] = None, **fields: Any) -> None:
    payload = _payload(event, level, logger_type, fields)
    _log_std(level, payload)
    # fire-and-forget only for sync paths
    if db_sink:
        asyncio.create_task(db_sink.enqueue_async(payload))  # best-effort in sync contexts


async def emit_async(event: str, *, level: LogLevel = LogLevel.INFO, logger_type: str = "decorator",
                     db_sink: Optional[AsyncDbSink] = None, **fields: Any) -> None:
    payload = _payload(event, level, logger_type, fields)
    _log_std(level, payload)
    if db_sink:
        await db_sink.enqueue_async(payload)  # durability in async contexts
