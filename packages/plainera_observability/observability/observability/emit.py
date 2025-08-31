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

from db_manager.sinks import DbSink
from .levels import LogLevel, STD_LEVEL
from .context import request_id_var
from .redact import scrub

logger = logging.getLogger("plainera")


def _submit_sink(db_sink: DbSink, payload: dict[str, Any]) -> None:
    """Submit the log payload to the configured database sink (best-effort).

    Invokes ``db_sink.enqueue(payload)``. If the sink returns a coroutine,
    it is scheduled via ``asyncio.create_task`` (fire-and-forget) so that
    log emission does not block the caller. Any exception raised during
    submission is caught and an internal error entry is logged instead.

    Args:
        db_sink (DbSink): The sink responsible for persisting log records.
        payload (dict[str, Any]): The fully constructed log payload to submit.

    Returns:
        None
    """
    try:
        rv = db_sink.enqueue(payload)
        if asyncio.iscoroutine(rv):
            asyncio.create_task(rv)  # fire-and-forget
    except Exception:
        # Never crash on sink errors; emit a minimal internal error entry.
        logger.log(
            STD_LEVEL[LogLevel.ERROR],
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "level": "error",
                    "event": "db_sink_enqueue_failed",
                    "logger_type": "system",
                    "request_id": request_id_var.get(),
                }
            ),
        )


def emit(
    event: str,
    *,
    level: LogLevel = LogLevel.INFO,
    logger_type: str = "decorator",
    db_sink: Optional[DbSink] = None,
    **fields: Any,
) -> None:
    """Emit a structured log entry and optionally forward it to a DB sink.

    Constructs a JSON-serializable payload with a UTC ``timestamp``, the textual
    ``level``, the ``event`` name, a ``logger_type`` tag, and the current
    ``request_id`` (from context var). Additional keyword ``fields`` are passed
    through a redaction/scrubbing step via ``scrub``. The payload is emitted to
    the module logger using the mapped standard level from ``STD_LEVEL``. If a
    ``db_sink`` is provided, the payload is also submitted asynchronously via
    :func:`_submit_sink`.

    This function must not raise; any sink-related errors are handled internally.

    Args:
        event (str): Short event name or message describing what happened.
        level (LogLevel): Severity level for the log entry. Defaults to ``INFO``.
        logger_type (str): Free-form classifier for the log source (e.g., "decorator",
            "middleware"). Defaults to "decorator".
        db_sink (Optional[DbSink]): Optional sink to persist/forward the payload.
        **fields (Any): Additional structured fields to include; values will be
            scrubbed/redacted as configured by ``scrub``.

    Returns:
        None
    """
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level.name.lower(),
        "event": event,
        "logger_type": logger_type,
        "request_id": request_id_var.get(),
        **scrub(fields),
    }
    logger.log(STD_LEVEL[level], json.dumps(payload, default=str))
    if db_sink:
        _submit_sink(db_sink, payload)
