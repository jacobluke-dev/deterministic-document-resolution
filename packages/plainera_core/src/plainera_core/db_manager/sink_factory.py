# plainera_core/db_manager/sink_factory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from public_api.db.models import Logger, PackageLogger  # adjust if paths differ
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
    key = (model, default_logger_type)
    mapper = _MAPPER_CACHE.get(key)
    if mapper is None:
        mapper = make_logger_mapper(model, default_logger_type=default_logger_type)
        _MAPPER_CACHE[key] = mapper
    return mapper

def available_sinks() -> list[str]:
    return sorted(_REGISTRY.keys())

def make_sink(sessionmaker: async_sessionmaker[AsyncSession], name: str) -> SqlAlchemyModelSink:
    """
    Async-only sink (use inside async code).
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
    """
    Sink usable from both async and sync call sites.
    """
    spec = _REGISTRY.get(name)
    if spec is None:
        raise ValueError(f"Unknown sink '{name}'. Valid: {', '.join(available_sinks())}")
    mapper = _mapper_for(spec.model, spec.default_logger_type)
    async_sink = SqlAlchemyModelSink(sessionmaker, spec.model, mapper)
    sync_sink = SyncSqlAlchemyModelSink(sync_url, spec.model, mapper)
    return UniversalSink(async_sink, sync_sink)

def register_sink(name: str, model: type[Any], default_logger_type: str = "decorator") -> None:
    _REGISTRY[name] = SinkSpec(model, default_logger_type)
    _MAPPER_CACHE.pop((model, default_logger_type), None)  # invalidate cached mapper
