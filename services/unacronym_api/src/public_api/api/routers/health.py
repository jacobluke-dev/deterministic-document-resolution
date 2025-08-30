from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy import text

from src.public_api.api.types import DBManagerDep
from src.public_api.db.migrations_status import is_at_head
from src.public_api.schemas.health_check import HealthCheckResponse

router = APIRouter()

@router.get("/healthz", summary="Liveness", response_model=HealthCheckResponse)
async def healthz() -> HealthCheckResponse:
    return HealthCheckResponse(status="ok")

@router.get("/readyz", response_model=HealthCheckResponse)
def readyz(dbm: DBManagerDep) -> HealthCheckResponse:
    with dbm.session() as s:
        s.execute(text("SELECT 1"))
        row = s.execute(text("SELECT 1 FROM alembic_version LIMIT 1")).fetchone()
        if not is_at_head(dbm.engine):
            raise HTTPException(status_code=503, detail="migrations not at head")
        if row is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                detail="migrations not applied")
    return HealthCheckResponse(status="ready")


@router.post("/_echo")
async def echo(request: Request) -> dict[str, int]:
    data = await request.body()
    return {"len": len(data)}
