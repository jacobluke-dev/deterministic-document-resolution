import asyncio
import inspect
import json
from functools import wraps
from time import monotonic
from typing import Callable, Iterable, Optional, Any

from .emit import emit
from .levels import LogLevel

_MISSING = object()

def _preview(value: Any, limit: int = 1024) -> str:
    """JSON-preview with fallback to repr, truncated to 'limit' chars."""
    try:
        s = json.dumps(value, default=str)
    except Exception:
        s = repr(value)
    if len(s) > limit:
        return s[:limit] + f"...(+{len(s)-limit} chars)"
    return s

def logger(message: str = "",
           *,
           arg_names: Optional[Iterable[str]] = None,
           redact: Optional[Iterable[str]] = None,  # explicit redaction; central scrub still applies
           log_result: bool = False,
           log_duration: bool = True,
           logger_type: str = "decorator",
           db_sink=None,
           result_max_len: int = 1024,
           result_transform: Optional[Callable[[Any], Any]] = None) -> Callable:
    arg_names = list(arg_names or [])
    redact = set(redact or [])

    def _select_args(func, args, kwargs):
        if not arg_names:
            return {}
        bound = inspect.signature(func).bind_partial(*args, **kwargs)
        bound.apply_defaults()
        picked = {k: bound.arguments.get(k, "<missing>") for k in arg_names}
        for k in redact:
            if k in picked:
                picked[k] = "[REDACTED]"
        return picked

    def decorate(func):
        is_coro = asyncio.iscoroutinefunction(func)

        @wraps(func)
        async def aw(*a, **kw):
            start = monotonic()
            res = _MISSING
            try:
                res = await func(*a, **kw)
                return res
            finally:
                duration_ms = int((monotonic() - start) * 1000) if log_duration else None
                fields = {
                    "function": func.__name__,
                    "duration_ms": duration_ms,
                    "args": _select_args(func, a, kw),
                }
                if log_result and res is not _MISSING:
                    try:
                        val = result_transform(res) if result_transform else res
                    except Exception:
                        val = "<result_transform_failed>"
                    fields["result"] = _preview(val, limit=result_max_len)
                emit(message or func.__name__, level=LogLevel.INFO, logger_type=logger_type, db_sink=db_sink, **fields)

        @wraps(func)
        def sw(*a, **kw):
            start = monotonic()
            res = _MISSING
            try:
                res = func(*a, **kw)
                return res
            finally:
                duration_ms = int((monotonic() - start) * 1000) if log_duration else None
                fields = {
                    "function": func.__name__,
                    "duration_ms": duration_ms,
                    "args": _select_args(func, a, kw),
                }
                if log_result and res is not _MISSING:
                    try:
                        val = result_transform(res) if result_transform else res
                    except Exception:
                        val = "<result_transform_failed>"
                    fields["result"] = _preview(val, limit=result_max_len)
                emit(message or func.__name__, level=LogLevel.INFO, logger_type=logger_type, db_sink=db_sink, **fields)

        return aw if is_coro else sw

    return decorate
