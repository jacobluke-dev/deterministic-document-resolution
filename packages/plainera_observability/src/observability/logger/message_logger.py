import inspect
import json
import logging
from typing import Any, Mapping, Optional

from .emit import emit
from .levels import LogLevel

LOG_NAME = "plainera"
log = logging.getLogger(LOG_NAME)

# library best practice
if not log.handlers:
    log.addHandler(logging.NullHandler())


def _to_text(v: Any, limit: int = 2048) -> Optional[str]:
    """
    Convert a value into a log-safe string, with optional truncation.

    Conversion rules:
      - None → returns None
      - bytes → UTF-8 decode (errors="replace")
      - str → unchanged
      - other → compact JSON via json.dumps(v, default=str, separators=(",", ":"))
        (falls back to str(v) if serialization fails)

    Truncation:
        If the resulting string exceeds `limit` characters, it is truncated to the
        first `limit` characters and suffixed with:
            ...(+N chars)
        where N is the number of omitted characters.

    Args:
        v (Any): Any input value to convert.
        limit (int): Maximum string length before truncation (default: 2048).

    Returns:
        str | None: A log-safe string representation or None.
    """
    if v is None:
        return None
    try:
        if isinstance(v, bytes):
            s = v.decode(errors="replace")
        elif isinstance(v, str):
            s = v
        else:
            # compact JSON (smaller logs) and stable formatting
            s = json.dumps(v, default=str, separators=(",", ":"))
    except Exception:
        s = str(v)
    return s if len(s) <= limit else s[:limit] + f"...(+{len(s)-limit} chars)"


def message_logger(
    message: str,
    level: LogLevel = LogLevel.INFO,
    *,
    logger_type: str = "inline",
    args: Optional[Mapping[str, Any]] = None,
    details: Optional[Mapping[str, Any] | str] = None,
    db_sink=None,
) -> None:
    """
    Ad-hoc structured logger.

    Mappings:
      - `message`  → `event`
      - `args`     → `args` (passed through; sensitive fields may be redacted by emit)
      - `details`  → `info` (stringified via `_to_text` for DB persistence)
      - `function` → auto-resolved caller name (via `inspect`)

    Behavior:
      - Emits a single structured log entry through `emit(...)`.
      - `logger_type` is included as a passthrough field (default: "inline").
      - `level` is converted to the normalized textual level by `emit`.
      - If provided, `db_sink` is forwarded to `emit` for DB write paths.

    Args:
        message: Event name to record.
        level: Log severity (default: `LogLevel.INFO`).
        logger_type: Logical channel/type for the log (e.g., "inline", "audit").
        args: Arbitrary key/value payload to attach under `args`.
        details: Additional rich info (mapping or string) serialized into `info`.
        db_sink: Optional sink used by `emit` for database persistence.

    Returns:
        None

    Notes:
        - The caller function name is captured without retaining frames (to avoid cycles).
        - `_to_text` handles string/bytes directly and JSON-serializes other objects,
          truncating long values with a `...(+N chars)` suffix.
        - Redaction (e.g., `authorization` → `[REDACTED]`) is applied inside `emit`.
    """
    frame = inspect.currentframe()
    try:
        caller = frame.f_back if frame else None
        func_name = caller.f_code.co_name if caller else None
    finally:
        # prevent reference cycles
        del frame

    emit(
        message,
        level=level,
        logger_type=f"message_logger.{logger_type}",
        db_sink=db_sink,
        function=func_name,
        args=dict(args) if args else None,
        info=_to_text(details),  # <-- map details -> info so it hits the DB
    )


# Optional convenience shorthands:
def info(message: str, **kw) -> None:
    message_logger(message, LogLevel.INFO, **kw)


def debug(message: str, **kw) -> None:
    message_logger(message, LogLevel.DEBUG, **kw)


def warning(message: str, **kw) -> None:
    message_logger(message, LogLevel.WARNING, **kw)


def error(message: str, **kw) -> None:
    message_logger(message, LogLevel.ERROR, **kw)
