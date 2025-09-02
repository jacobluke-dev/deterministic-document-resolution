import asyncio, json, logging
from datetime import datetime, timezone
from typing import Any, Optional

from .levels import LogLevel, STD_LEVEL
from .context import request_id_var
from .redact import scrub
from ..core.types import SyncSink, AsyncSink

logger = logging.getLogger("plainera")

def _make_payload(event: str, level: LogLevel | int | str, logger_type: str, **fields: Any) -> dict[str, Any]:
    lvl = level.name.lower() if hasattr(level, "name") else (level if isinstance(level, int) else str(level).lower())
    return scrub({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": lvl,
        "event": event,
        "logger_type": logger_type,
        "request_id": request_id_var.get(),
        **fields,
    })

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

    if db_sink is not None:
        if hasattr(db_sink, "enqueue"):
            # Pure sync sink
            db_sink.enqueue(payload)  # type: ignore[attr-defined]
        elif hasattr(db_sink, "enqueue_async"):
            # Sink is async; run it to completion (blocking this thread)
            asyncio.run(db_sink.enqueue_async(payload))  # type: ignore[attr-defined]
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

    if db_sink is not None:
        if hasattr(db_sink, "enqueue_async"):
            await db_sink.enqueue_async(payload)  # type: ignore[attr-defined]
        elif hasattr(db_sink, "enqueue"):
            # Sink is sync; run it in a thread so we don't block the loop
            await asyncio.to_thread(db_sink.enqueue, payload)  # type: ignore[attr-defined]
    logger.log(STD_LEVEL[LogLevel(level) if isinstance(level, int) else level], json.dumps(payload))
