from __future__ import annotations

import inspect
from typing import Any, Mapping, Optional

from .emit import emit
from .levels import LogLevel

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
    Ad-hoc, structured event logger

    - `message` becomes the structured `event`
    - `args` is captured as `args` (redaction happens centrally in emit)
    - `details` is an optional free-form field (dict or str)
    - `function` is auto-populated from the caller
    """
    frame = inspect.currentframe()
    caller = frame.f_back if frame else None
    func_name = caller.f_code.co_name if caller else None

    emit(
        message,
        level=level,
        logger_type=logger_type,
        db_sink=db_sink,
        function=func_name,
        args=dict(args) if args else None,
        details=details,
    )

# Optional convenience shorthands:
def info(message: str, **kw) -> None:    message_logger(message, LogLevel.INFO, **kw)
def debug(message: str, **kw) -> None:   message_logger(message, LogLevel.DEBUG, **kw)
def warning(message: str, **kw) -> None: message_logger(message, LogLevel.WARNING, **kw)
def error(message: str, **kw) -> None:   message_logger(message, LogLevel.ERROR, **kw)
