# services/public_api/src/public_api/api/types.py
from __future__ import annotations

from asyncio import Semaphore
from collections.abc import Awaitable, Iterable
from typing import Annotated, Optional, Protocol, TypeAlias, TypedDict

from plainera_core.db_manager.connection import DBManager
from fastapi import Depends, Header

from public_api.core import deps  # get_resolver / get_semaphore

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


ResolverT: TypeAlias = ResolverProtocol


# --- Common DI aliases -------------------------------------------------------

ResolverDep: TypeAlias = Annotated[ResolverT, Depends(deps.get_resolver)]
SemaphoreDep: TypeAlias = Annotated[Semaphore | None, Depends(deps.get_semaphore)]
DBManagerDep: TypeAlias = Annotated[DBManager, Depends(deps.get_dbm)]


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
