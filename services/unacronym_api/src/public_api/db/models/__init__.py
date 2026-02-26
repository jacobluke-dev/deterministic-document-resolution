# services/unacronym_api/src/public_api/db/models/__init__.py

from observability.db.models.base import Base, BaseWithTimestamps
from plainera_unacronym.db.models.logger import PackageLogger

from .acronym_alias import AcronymAlias
from .api_key import ApiKey

# Eager imports so SQLAlchemy sees everything in the registry
from .glossary_entry import GlossaryEntry
from .logger import Logger

__all__ = [
    "ApiKey",
    "Base",
    "BaseWithTimestamps",
    "GlossaryEntry",
    "AcronymAlias",
    "Logger",
    "PackageLogger",
]
