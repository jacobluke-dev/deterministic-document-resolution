import asyncio
from typing import Any, Awaitable, Callable, Protocol

from sqlalchemy import insert

# ---- Protocol ---------------------------------------------------------------

class DbSink(Protocol):
    def enqueue(self, payload: dict[str, Any]) -> None | Awaitable[None]: ...

Predicate = Callable[[dict[str, Any]], bool]
MapPayload = Callable[[dict[str, Any]], dict[str, Any]]

# ---- Single-model sink (async SQLAlchemy) -----------------------------------


class AsyncDbSink(Protocol):
    async def enqueue_async(self, payload: dict[str, Any]) -> None: ...

class SqlAlchemyModelSink:
    def __init__(self, session_factory, model_cls, mapper):
        self._sf = session_factory
        self._model = model_cls
        self._map = mapper

    async def enqueue_async(self, payload: dict[str, Any]) -> None:
        row = self._map(payload)
        async with self._sf() as s:
            await s.execute(insert(self._model).values(**row))
            await s.commit()

# ---- Fan-out and routing ----------------------------------------------------


class CompositeSink:
    """Fan-out sink that forwards log payloads to multiple downstream sinks.

    This sink is useful when you want each log record to be persisted or
    transmitted to more than one backend. For example, you might want to
    write the same structured log payload to both a SQL database and
    standard output.

    Each downstream sink is invoked with the same payload. If a sink's
    ``enqueue`` method returns a coroutine, it is scheduled via
    ``asyncio.create_task`` in a fire-and-forget manner, so failures or
    shutdown timing can cause payload loss in short-lived processes.

    Example:
        >>> sink = CompositeSink([SqlAlchemyModelSink(...), ConsoleSink()])
        >>> sink.enqueue({"event": "user_signup", "level": "info"})

    Args:
        sinks (Iterable[DbSink]): A sequence of sink instances that implement
            an ``enqueue`` method accepting a log payload.
    """

    def __init__(self):
        self._sinks = {}

    def enqueue(self, payload: dict[str, Any]) -> None:
        """Forward a log payload to all configured sinks.

        Args:
            payload (dict[str, Any]): The structured log payload to send.

        Notes:
            - If a downstream sink's ``enqueue`` returns a coroutine,
              this method schedules it on the event loop using
              ``asyncio.create_task``.
            - Fire-and-forget means the caller does not wait for all
              downstream sinks to finish writing.
        """
        for s in self._sinks:
            rv = s.enqueue(payload)
            if asyncio.iscoroutine(rv):
                asyncio.create_task(rv)



class RouterSink:
    """Routing sink that forwards log payloads only to selected sinks.

    Each sink is associated with a predicate function. When a payload is
    enqueued, the predicate is evaluated; if it returns ``True``, the
    payload is forwarded to that sink. This allows fine-grained routing
    based on log content (e.g., error-level events to the DB, access logs
    to a file).

    Example:
        >>> error_only = lambda p: p.get("level") == "error"
        >>> info_only = lambda p: p.get("level") == "info"
        >>> sink = RouterSink([
        ...     (error_only, SqlAlchemyModelSink(...)),
        ...     (info_only, ConsoleSink()),
        ... ])
        >>> sink.enqueue({"event": "startup", "level": "info"})
        # goes only to ConsoleSink

    Args:
        routes (list[tuple[Predicate, DbSink]]): A list of (predicate, sink)
            pairs. ``predicate`` is a callable that takes a payload dict
            and returns a boolean. ``sink`` is any sink with an ``enqueue``
            method.
    """

    def __init__(self):
        self._routes = {}

    def enqueue(self, payload: dict[str, Any]) -> None:
        """Forward a log payload to sinks whose predicates match.

        Args:
            payload (dict[str, Any]): The structured log payload to evaluate
                against routing predicates.

        Notes:
            - Multiple predicates can match, in which case the payload is
              sent to multiple sinks.
            - If a downstream sink's ``enqueue`` returns a coroutine,
              it is scheduled with ``asyncio.create_task`` (fire-and-forget).
        """
        for pred, sink in self._routes:
            if pred(payload):
                rv = sink.enqueue(payload)
                if asyncio.iscoroutine(rv):
                    asyncio.create_task(rv)
