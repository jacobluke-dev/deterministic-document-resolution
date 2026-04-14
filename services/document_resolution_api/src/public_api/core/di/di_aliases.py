from asyncio import Semaphore
from typing import Annotated, TypeAlias

from fastapi import Depends
from document_resolution_core.db_manager.connection import DBManager

from public_api.core.di import deps

SemaphoreDep: TypeAlias = Annotated[Semaphore | None, Depends(deps.get_semaphore)]
DBManagerDep: TypeAlias = Annotated[DBManager, Depends(deps.get_dbm)]
