from typing import TYPE_CHECKING, Any
import importlib as _importlib

__all__ = (
    "Base",
    "BaseWithTimestamps",
    "GlossaryEntry",
    "AcronymAlias",
    "Logger",
)

if TYPE_CHECKING:
    from .base import Base, BaseWithTimestamps
    from .glossary_entry import GlossaryEntry
    from .acronym_alias import AcronymAlias
    from .logger import Logger

_lazy_attrs: dict[str, tuple[str, str]] = {
    "Base": (".base", "Base"),
    "BaseWithTimestamps": (".base", "BaseWithTimestamps"),
    "GlossaryEntry": (".glossary_entry", "GlossaryEntry"),
    "AcronymAlias": (".acronym_alias", "AcronymAlias"),
    "Logger": (".logger", "Logger"),
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

# Ensure mapped classes are registered so string relationships resolve
from .glossary_entry import GlossaryEntry as _GE  # noqa: F401
from .acronym_alias import AcronymAlias as _AA    # noqa: F401
