import asyncio, inspect
from functools import wraps
from time import monotonic
from typing import Callable, Iterable, Optional
from .emit import emit
from .levels import LogLevel

def logger(message: str = "",
           *,
           arg_names: Optional[Iterable[str]] = None,
           redact: Optional[Iterable[str]] = None,  # handled centrally; here we just pass args
           log_result: bool = False,
           log_duration: bool = True,
           logger_type: str = "decorator",
           db_sink=None) -> Callable:
    arg_names = list(arg_names or [])
    redact = set(redact or [])
    def _select_args(func, args, kwargs):
        if not arg_names:
            return {}
        bound = inspect.signature(func).bind_partial(*args, **kwargs)
        bound.apply_defaults()
        picked = {k: bound.arguments.get(k, "<missing>") for k in arg_names}
        # Keep keys; redact logic is in emit.scrub, but allow explicit blanks here too
        for k in redact:
            if k in picked: picked[k] = "[REDACTED]"
        return picked

    def decorate(func):
        is_coro = asyncio.iscoroutinefunction(func)
        @wraps(func)
        async def aw(*a, **kw):
            start = monotonic()
            try:
                res = await func(*a, **kw)
                return res
            finally:
                duration_ms = int((monotonic() - start) * 1000) if log_duration else None
                emit(
                    message or func.__name__,
                    level=LogLevel.INFO,
                    logger_type=logger_type,
                    db_sink=db_sink,
                    function=func.__name__,
                    duration_ms=duration_ms,
                    args=_select_args(func, a, kw),
                )
        @wraps(func)
        def sw(*a, **kw):
            start = monotonic()
            try:
                res = func(*a, **kw)
                return res
            finally:
                duration_ms = int((monotonic() - start) * 1000) if log_duration else None
                emit(
                    message or func.__name__,
                    level=LogLevel.INFO,
                    logger_type=logger_type,
                    db_sink=db_sink,
                    function=func.__name__,
                    duration_ms=duration_ms,
                    args=_select_args(func, a, kw),
                )
        return aw if is_coro else sw
    return decorate
