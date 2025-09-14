import importlib as _importlib
from typing import TYPE_CHECKING, Any

__all__ = (
    "Base",
    "BaseWithTimestamps",
    "GlossaryEntry",
    "AcronymAlias",
    "Logger",
    "PackageLogger",
)

if TYPE_CHECKING:
    from .acronym_alias import AcronymAlias
    from observability.db.models.base import BaseWithTimestamps, Base
    from .glossary_entry import GlossaryEntry
    from .logger import Logger
    from plainera_unacronym.db.models.logger import PackageLogger

_lazy_attrs: dict[str, tuple[str, str]] = {
    "Base": (".base", "Base"),
    "BaseWithTimestamps": (".base", "BaseWithTimestamps"),
    "GlossaryEntry": (".glossary_entry", "GlossaryEntry"),
    "AcronymAlias": (".acronym_alias", "AcronymAlias"),
    "Logger": (".logger", "Logger"),
    "PackageLogger": (".package_logger", "PackageLogger"),
}

def __getattr__(name: str) -> Any:
    try:
        mod_rel, attr = _lazy_attrs[name]
    except KeyError as e:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from e
    module = _importlib.import_module(mod_rel, __name__)
    value = getattr(module, attr)
    globals()[name] = value
    return value

def __dir__() -> list[str]:
    # So tab-completion shows lazies too
    return sorted(list(globals().keys()) + list(_lazy_attrs.keys()))
