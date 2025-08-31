import asyncio
import inspect
import json
from functools import wraps
from time import monotonic
from typing import Any, Callable, Iterable, Optional, ParamSpec, TypeVar

from .emit import emit
from .levels import LogLevel

_MISSING = object()
P = ParamSpec("P")
R = TypeVar("R")


def _preview(value: Any, limit: int = 1024) -> str:
    """Return a JSON-safe string preview of a value, truncated to ``limit`` characters.

    Attempts ``json.dumps(value, default=str)``; falls back to ``repr(value)`` if
    JSON serialization fails. If the resulting string exceeds ``limit`` characters,
    it is truncated and suffixed with ``"...(+N chars)"``.

    Args:
        value (Any): Value to serialize for logging.
        limit (int): Maximum number of characters to include in the preview.

    Returns:
        str: Serialized (and possibly truncated) representation of ``value``.
    """
    try:
        s = json.dumps(value, default=str)
    except Exception:
        s = repr(value)
    return s if len(s) <= limit else s[:limit] + f"...(+{len(s)-limit} chars)"


def logger(
    message: str = "",
    *,
    arg_names: Optional[Iterable[str]] = None,
    redact: Optional[Iterable[str]] = None,
    log_result: bool = False,
    log_duration: bool = True,
    log_before: bool = False,
    logger_type: str = "decorator",
    db_sink=None,  # type: ignore[assignment]  # keep loose to avoid circular import
    on_error_level: LogLevel | str = LogLevel.ERROR,
    result_max_len: int = 1024,
    result_transform: Optional[Callable[[Any], Any]] = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator factory for structured logging around sync/async callables.

    Emits a log entry on success (INFO) and on exception (``on_error_level``). Can
    record selected call arguments, a preview of the return value, and the execution
    duration. Optionally emits a pre-call entry.

    The wrapper preserves coroutine behavior for ``async def`` functions.

    Args:
        message (str): Log message. Defaults to the wrapped function's name if empty.
        arg_names (Optional[Iterable[str]]): Parameter names to capture under the
            ``args`` field. Omitted if not provided.
        redact (Optional[Iterable[str]]): Subset of ``arg_names`` whose values will be
            replaced with ``"[REDACTED]"``.
        log_result (bool): If True, include a preview of the return value under
            ``result`` using ``_preview`` (after ``result_transform`` if provided).
        log_duration (bool): If True, include ``duration_ms`` (integer milliseconds).
        log_before (bool): If True, emit an INFO entry before invoking the function.
        logger_type (str): Free-form classifier passed through to the sink (e.g., "decorator").
        db_sink: Optional sink callable accepted by ``emit`` for persistence/forwarding.
        on_error_level (LogLevel | str): Log level used when an exception occurs.
        result_max_len (int): Maximum characters to include in the result preview.
        result_transform (Optional[Callable[[Any], Any]]): Optional transformation
            applied to the function's result before previewing/logging.

    Returns:
        Callable[[Callable[P, R]], Callable[P, R]]: A decorator that wraps the target callable.

    Raises:
        Exception: Re-raises any exception from the wrapped callable after logging it.

    Logged fields:
        - ``function``: The wrapped function's ``__name__``.
        - ``args``: Dict of selected arguments (from ``arg_names``; redacted as configured).
        - ``duration_ms``: Included when ``log_duration`` is True.
        - ``result``: Included when ``log_result`` is True.
        - ``error``: Included on exceptions (stringified).
    """
    names = list(arg_names or [])
    redact_set = set(redact or [])

    def _select_args(func, args, kwargs) -> dict[str, Any]:
        if not names:
            return {}
        bound = inspect.signature(func).bind_partial(*args, **kwargs)
        bound.apply_defaults()
        picked = {k: bound.arguments.get(k, "<missing>") for k in names}
        for k in redact_set:
            if k in picked:
                picked[k] = "[REDACTED]"
        return picked

    def _level_norm(level: LogLevel | str) -> LogLevel | str:
        return level  # keep passthrough; emit accepts LogLevel

    def _finalize(level: LogLevel | str, res: Any, start: float, func, args, kwargs) -> None:
        duration_ms = int((monotonic() - start) * 1000) if log_duration else None
        fields: dict[str, Any] = {
            "function": func.__name__,
            "args": _select_args(func, args, kwargs),
        }
        if duration_ms is not None:
            fields["duration_ms"] = duration_ms
        if log_result and res is not _MISSING:
            try:
                val = result_transform(res) if result_transform else res
            except Exception:
                val = "<result_transform_failed>"
            fields["result"] = _preview(val, limit=result_max_len)
        emit(message or func.__name__, level=_level_norm(level), logger_type=logger_type, db_sink=db_sink, **fields)

    def decorate(func: Callable[P, R]) -> Callable[P, R]:
        is_coro = asyncio.iscoroutinefunction(func)

        if is_coro:
            @wraps(func)
            async def aw(*a: P.args, **kw: P.kwargs) -> R:
                if log_before:
                    emit(
                        message or "Executing function",
                        level=LogLevel.INFO,
                        logger_type=logger_type,
                        db_sink=db_sink,
                        function=func.__name__,
                        args=_select_args(func, a, kw),
                    )
                start = monotonic()
                try:
                    res = await func(*a, **kw)
                    _finalize(LogLevel.INFO, res, start, func, a, kw)
                    return res
                except Exception as e:
                    emit(
                        f"Exception in {func.__name__}",
                        level=_level_norm(on_error_level),
                        logger_type=logger_type,
                        db_sink=db_sink,
                        function=func.__name__,
                        args=_select_args(func, a, kw),
                        error=repr(e),
                    )
                    raise
            return aw  # type: ignore[return-value]
        else:
            @wraps(func)
            def sw(*a: P.args, **kw: P.kwargs) -> R:
                if log_before:
                    emit(
                        message or "Executing function",
                        level=LogLevel.INFO,
                        logger_type=logger_type,
                        db_sink=db_sink,
                        function=func.__name__,
                        args=_select_args(func, a, kw),
                    )
                start = monotonic()
                try:
                    res = func(*a, **kw)
                    _finalize(LogLevel.INFO, res, start, func, a, kw)
                    return res
                except Exception as e:
                    emit(
                        f"Exception in {func.__name__}",
                        level=_level_norm(on_error_level),
                        logger_type=logger_type,
                        db_sink=db_sink,
                        function=func.__name__,
                        args=_select_args(func, a, kw),
                        error=repr(e),
                    )
                    raise
            return sw  # type: ignore[return-value]
    return decorate
