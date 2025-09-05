# services/unacronym_api/src/public_api/core/di_aliases.py
from asyncio import Semaphore
from typing import Annotated, TypeAlias

from fastapi import Depends
from plainera_core.db_manager.connection import DBManager

from public_api.core import deps
from public_api.core.providers import AcronymResolverLike as ResolverT

ResolverDep: TypeAlias = Annotated[ResolverT, Depends(deps.get_resolver)]
SemaphoreDep: TypeAlias = Annotated[Semaphore | None, Depends(deps.get_semaphore)]
DBManagerDep: TypeAlias = Annotated[DBManager, Depends(deps.get_dbm)]
