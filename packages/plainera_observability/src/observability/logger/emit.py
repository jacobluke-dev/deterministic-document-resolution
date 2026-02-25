import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from observability.core.types import AsyncSink, SyncSink

from .context import request_id_var
from .levels import STD_LEVEL, LogLevel
from .redact import scrub

logger = logging.getLogger("plainera")


def _make_payload(event: str, level: LogLevel | int | str, logger_type: str, **fields: Any) -> dict[str, Any]:
    lvl = level.name.lower() if hasattr(level, "name") else (level if isinstance(level, int) else str(level).lower())
    return scrub(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": lvl,
            "event": event,
            "logger_type": logger_type,
            "request_id": request_id_var.get(),
            **fields,
        }
    )


def emit(
    event: str,
    *,
    level: LogLevel = LogLevel.INFO,
    logger_type: str = "decorator",
    db_sink: Optional[SyncSink | AsyncSink] = None,
    **fields: Any,
) -> None:
    """Synchronous: completes the DB write before returning. Use in sync code."""
    payload = _make_payload(event, level, logger_type, **fields)

    try:
        if db_sink is not None:
            if hasattr(db_sink, "enqueue"):
                db_sink.enqueue(payload)
            elif hasattr(db_sink, "enqueue_async"):
                asyncio.run(db_sink.enqueue_async(payload))
    except Exception as e:
        # Never let logging break the app
        logger.warning("db_sink failed: %r", e)

    logger.log(STD_LEVEL[LogLevel(level) if isinstance(level, int) else level], json.dumps(payload))


async def emit_async(
    event: str,
    *,
    level: LogLevel = LogLevel.INFO,
    logger_type: str = "decorator",
    db_sink: Optional[SyncSink | AsyncSink] = None,
    **fields: Any,
) -> None:
    """Asynchronous: await the DB write. Use inside async functions."""
    payload = _make_payload(event, level, logger_type, **fields)

    try:
        if db_sink is not None:
            if hasattr(db_sink, "enqueue_async"):
                await db_sink.enqueue_async(payload)
            elif hasattr(db_sink, "enqueue"):
                await asyncio.to_thread(db_sink.enqueue, payload)
    except Exception as e:
        logger.warning("db_sink failed: %r", e)

    logger.log(STD_LEVEL[LogLevel(level) if isinstance(level, int) else level], json.dumps(payload))
