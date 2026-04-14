# services/public_api/src/public_api/api/types.py
from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from typing import Annotated, Optional, Protocol, TypeAlias, TypedDict

from fastapi import Header
from document_resolution_core.db_manager.connection import DBManager

# --- Structural contracts (no core imports in annotations) -------------------

class AcronymLike(Protocol):
    """
    Minimal shape we need from an Acronym.
    """
    text: str


class DefinitionCandidateLike(Protocol):
    """
    Minimal shape we need from a DefinitionCandidate.
    """
    text: str
    score: float


class ResolverProtocol(Protocol):
    """
    Minimal protocol the resolver must satisfy.
    """
    def resolve(
        self,
        acro: AcronymLike,
        top_k: int = 5,
    ) -> Iterable[DefinitionCandidateLike] | Awaitable[Iterable[DefinitionCandidateLike]]:
        ...

# Common headers
RequestIdHeader: TypeAlias = Annotated[
    Optional[str],
    Header(default=None, convert_underscores=False),
]


class APIDefinition(TypedDict):
    text: str
    start: int
    end: int
    confidence: float
    source: str

class AppState(Protocol):
    """
    Providing the DB Manager to the FastAPI class.
    """
    dbm: DBManager

"""Type alias for a synchronous lookup function.

Args:
    str: The acronym text to resolve.

Returns:
    Iterable[DefinitionCandidateLike]: A collection of candidate definitions.
"""
LookupFunc: TypeAlias = Callable[[str], Iterable[DefinitionCandidateLike]]
"""Type alias for a resolver return type.

The resolver may be synchronous or asynchronous.

Returns:
    Iterable[DefinitionCandidateLike] | Awaitable[Iterable[DefinitionCandidateLike]]:
        Candidate definitions either directly or wrapped in an awaitable.
"""
ResolveReturn: TypeAlias = Iterable[DefinitionCandidateLike] | Awaitable[Iterable[DefinitionCandidateLike]]
