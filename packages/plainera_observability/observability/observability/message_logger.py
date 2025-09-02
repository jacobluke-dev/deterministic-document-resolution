import inspect
import json
from typing import Any, Mapping, Optional

from .emit import emit
from .levels import LogLevel

def _to_text(v: Any, limit: int = 2048) -> Optional[str]:
    if v is None:
        return None
    try:
        s = json.dumps(v, default=str)
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

    - `message` -> event
    - `args` -> captured as `args` (payload key)
    - `details` -> mapped to `info` (stringified) so it persists with current mapper
    - `function` -> auto-resolved from caller
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
        logger_type=logger_type,
        db_sink=db_sink,
        function=func_name,
        args=dict(args) if args else None,
        info=_to_text(details),  # <-- map details -> info so it hits the DB
    )

# Optional convenience shorthands:
def info(message: str, **kw) -> None:    message_logger(message, LogLevel.INFO, **kw)
def debug(message: str, **kw) -> None:   message_logger(message, LogLevel.DEBUG, **kw)
def warning(message: str, **kw) -> None: message_logger(message, LogLevel.WARNING, **kw)
def error(message: str, **kw) -> None:   message_logger(message, LogLevel.ERROR, **kw)
