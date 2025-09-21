from dataclasses import dataclass
from typing import Any, Callable

from plainera_unacronym.db.models.logger import PackageLogger
from public_api.db.models import Logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .mappers import make_logger_mapper
from .sinks import SqlAlchemyModelSink, SyncSqlAlchemyModelSink, UniversalSink

MapperFn = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class SinkSpec:
    model: type[Any]
    default_logger_type: str


_REGISTRY: dict[str, SinkSpec] = {
    "logger": SinkSpec(Logger, "api"),
    "package_logger": SinkSpec(PackageLogger, "package"),
}

# --- mapper cache (avoids lru_cache Hashable typing issues) -------------------
_MAPPER_CACHE: dict[tuple[type[Any], str], MapperFn] = {}


def _mapper_for(model: type[Any], default_logger_type: str) -> MapperFn:
    """Return a cached/constructed payload→row mapper for a given model and
    logger type.

    Uses an internal cache keyed by ``(model, default_logger_type)``. If a mapper
    does not exist for the key, it is created via ``make_logger_mapper`` and stored.

    Args:
        model (type[Any]): SQLAlchemy mapped class the mapper targets.
        default_logger_type (str): Fallback ``logger_type`` applied by the mapper
            when the payload omits one.

    Returns:
        MapperFn: Callable that transforms a logging payload (``dict[str, Any]``)
        into a model-aligned row (``dict[str, Any]``).

    Notes:
        - Cache key is the identity of ``model`` and the exact ``default_logger_type`` string.
        - This cache is in-memory and not synchronized across processes.
    """
    key = (model, default_logger_type)
    mapper = _MAPPER_CACHE.get(key)
    if mapper is None:
        mapper = make_logger_mapper(model, default_logger_type=default_logger_type)
        _MAPPER_CACHE[key] = mapper
    return mapper


def available_sinks() -> list[str]:
    """Return the list of registered sink names in sorted order.

    Returns:
        list[str]: Alphabetically sorted keys from the internal sink registry.
    """
    return sorted(_REGISTRY.keys())


def make_sink(sessionmaker: async_sessionmaker[AsyncSession], name: str) -> SqlAlchemyModelSink:
    """Create an async-only SQLAlchemy sink for a registered logger.

    Args:
        sessionmaker (async_sessionmaker[AsyncSession]): Async SQLAlchemy session
            factory used by the async sink.
        name (str): Registered sink name to resolve (e.g., ``"logger"``).

    Returns:
        SqlAlchemyModelSink: Async sink configured for the registry entry’s model
        with a cached/constructed mapper.

    Raises:
        ValueError: If ``name`` is not found in the internal registry.

    Notes:
        The mapper is retrieved from an internal cache keyed by
        ``(model, default_logger_type)``; if absent, it is constructed and cached.
    """
    spec = _REGISTRY.get(name)
    if spec is None:
        raise ValueError(f"Unknown sink '{name}'. Valid: {', '.join(available_sinks())}")
    mapper = _mapper_for(spec.model, spec.default_logger_type)
    return SqlAlchemyModelSink(sessionmaker, spec.model, mapper)


def make_universal_sink(
    sessionmaker: async_sessionmaker[AsyncSession],
    sync_url: str,
    name: str,
) -> UniversalSink:
    """Create a sink usable from both async and sync call sites.

    Looks up the sink specification by ``name`` in the internal registry,
    builds (or reuses, via cache) the mapper for the spec’s ``(model, default_logger_type)``,
    and returns a ``UniversalSink`` that wraps:
      - an async ``SqlAlchemyModelSink`` (using the provided ``sessionmaker``), and
      - a sync ``SyncSqlAlchemyModelSink`` (using ``sync_url``),
    both targeting the same model and mapper.

    Args:
        sessionmaker (async_sessionmaker[AsyncSession]): Async SQLAlchemy session factory for the async sink.
        sync_url (str): SQLAlchemy database URL for the sync sink.
        name (str): Registered sink name to resolve (e.g., ``"logger"``).

    Returns:
        UniversalSink: A wrapper exposing both ``enqueue_async`` and ``enqueue`` via
        its underlying async and sync sinks.

    Raises:
        ValueError: If ``name`` is not found in the registry.

    Notes:
        The async and sync sinks share the same mapper instance to ensure consistent
        row-shaping across execution contexts.
    """
    spec = _REGISTRY.get(name)
    if spec is None:
        raise ValueError(f"Unknown sink '{name}'. Valid: {', '.join(available_sinks())}")
    mapper = _mapper_for(spec.model, spec.default_logger_type)
    async_sink = SqlAlchemyModelSink(sessionmaker, spec.model, mapper)
    sync_sink = SyncSqlAlchemyModelSink(sync_url, spec.model, mapper)
    print("HELLLO!!! SINK, ", async_sink, sync_sink)
    return UniversalSink(async_sink, sync_sink)


def register_sink(name: str, model: type[Any], default_logger_type: str = "decorator") -> None:
    """Register (or overwrite) a sink specification and invalidate its cached
    mapper.

    Adds/updates an entry in the internal registry mapping ``name`` → ``SinkSpec(model, default_logger_type)``.
    If a mapper for the exact ``(model, default_logger_type)`` tuple exists in the cache, it is removed so that a
    fresh mapper will be created on next use.

    Args:
        name: Unique key used to reference the sink (e.g., ``"logger"``).
        model: SQLAlchemy mapped class associated with the sink.
        default_logger_type: Fallback ``logger_type`` to apply when the payload omits one.
            Defaults to ``"decorator"``.

    Returns:
        None

    Side Effects:
        - Mutates the internal sink registry.
        - Removes a single matching entry from the mapper cache for ``(model, default_logger_type)``.

    Notes:
        Overwriting an existing sink name does not purge any *previous* mapper cache entries that
        referenced a different ``(model, default_logger_type)`` pair.
    """
    _REGISTRY[name] = SinkSpec(model, default_logger_type)
    _MAPPER_CACHE.pop((model, default_logger_type), None)  # invalidate cached mapper
