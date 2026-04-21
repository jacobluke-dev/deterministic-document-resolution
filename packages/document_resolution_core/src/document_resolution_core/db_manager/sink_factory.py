import os
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .mappers import make_logger_mapper
from .sessions import make_async_session_maker, to_asyncpg
from .sinks import SqlAlchemyModelSink, SyncSqlAlchemyModelSink, UniversalSink
from .types import MapperFn


@dataclass(frozen=True)
class SinkSpec:
    model: type[Any]
    default_logger_type: str


_REGISTRY: dict[str, SinkSpec] = {}

# --- mapper cache (avoids lru_cache Hashable typing issues) -------------------
_MAPPER_CACHE: dict[tuple[type[Any], str], MapperFn] = {}


def _mapper_for(model: type[Any], default_logger_type: str) -> MapperFn:
    """Return a cached/constructed payload→row mapper for a given model and
    logger type.

    Args:
        model (type[Any]): SQLAlchemy mapped class the mapper targets.
        default_logger_type (str): Fallback ``logger_type`` applied by the mapper
            when the payload omits one.

    Returns:
        MapperFn: Callable that transforms a logging payload (``dict[str, Any]``)
        into a model-aligned row (``dict[str, Any]``).
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

    Args:
        sessionmaker (async_sessionmaker[AsyncSession]): Async SQLAlchemy session factory for the async sink.
        sync_url (str): SQLAlchemy database URL for the sync sink.
        name (str): Registered sink name to resolve (e.g., ``"logger"``).

    Returns:
        UniversalSink: A wrapper exposing both ``enqueue_async`` and ``enqueue`` via
        its underlying async and sync sinks.
    """
    spec = _REGISTRY.get(name)
    if spec is None:
        raise ValueError(f"Unknown sink '{name}'. Valid: {', '.join(available_sinks())}")
    mapper = _mapper_for(spec.model, spec.default_logger_type)
    async_sink = SqlAlchemyModelSink(sessionmaker, spec.model, mapper)
    sync_sink = SyncSqlAlchemyModelSink(sync_url, spec.model, mapper)
    return UniversalSink(async_sink, sync_sink)


def register_sink(name: str, model: type[Any], default_logger_type: str = "decorator") -> None:
    """Register (or overwrite) a sink specification and invalidate its cached
    mapper.

    Args:
        name: Unique key used to reference the sink (e.g., ``"logger"``).
        model: SQLAlchemy mapped class associated with the sink.
        default_logger_type: Fallback ``logger_type`` to apply when the payload omits one.
            Defaults to ``"decorator"``.

    Returns:
        None
    """
    _REGISTRY[name] = SinkSpec(model, default_logger_type)
    _MAPPER_CACHE.pop((model, default_logger_type), None)  # invalidate cached mapper


def build_sink_from_env(name: str) -> UniversalSink:
    """Build a universal database sink from environment configuration.

    This reads synchronous and asynchronous database URLs from environment
    variables and constructs a ``UniversalSink`` for the named sink target.

    Args:
      name: Logical sink name used when constructing the universal sink.

    Returns:
      A configured ``UniversalSink`` using an async session factory and sync
      database URL.

    Raises:
      RuntimeError: If neither the required sync nor async database URL
        configuration can be resolved from the environment.
    """
    sync_url = os.getenv("DATABASE_URL_SYNC") or os.getenv("TEST_DB_URL")
    async_url = os.getenv("DATABASE_URL") or (to_asyncpg(sync_url) if sync_url else None)

    if not (sync_url and async_url):
        raise RuntimeError(
            "Database URLs not set. Provide DATABASE_URL and DATABASE_URL_SYNC "
            "or a TEST_DB_URL."
        )

    session_local = make_async_session_maker(async_url)
    return make_universal_sink(session_local, sync_url, name)
