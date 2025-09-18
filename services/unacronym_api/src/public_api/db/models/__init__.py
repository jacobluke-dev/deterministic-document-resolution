# services/unacronym_api/src/public_api/db/models/__init__.py

from observability.db.models.base import Base, BaseWithTimestamps

# Eager imports so SQLAlchemy sees everything in the registry
from .glossary_entry import GlossaryEntry
from .acronym_alias import AcronymAlias
from .logger import Logger
from plainera_unacronym.db.models.logger import PackageLogger  # if you need this here

__all__ = [
    "Base",
    "BaseWithTimestamps",
    "GlossaryEntry",
    "AcronymAlias",
    "Logger",
    "PackageLogger",
]
