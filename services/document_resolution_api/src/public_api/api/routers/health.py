from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from public_api.core.di.di_aliases import DBManagerDep
from public_api.core.settings import db_settings
from public_api.db.migrations_status import is_at_head
from public_api.schemas.health_check import HealthCheckResponse

router = APIRouter()

@router.get("/healthz", summary="Liveness", response_model=HealthCheckResponse)
async def healthz() -> HealthCheckResponse:
    return HealthCheckResponse(status="ok")

@router.get("/readyz", response_model=HealthCheckResponse)
def readyz(dbm: DBManagerDep) -> HealthCheckResponse:
    schema = db_settings.DB_SCHEMA
    try:
        with dbm.session() as s:
            s.execute(text("SELECT 1"))
            s.execute(text(f'SELECT 1 FROM "{schema}".alembic_version LIMIT 1'))
        at_head = is_at_head(dbm.engine, schema=schema)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"readiness check failed: {exc}") from exc

    if not at_head:
        raise HTTPException(status_code=503, detail="migrations not at head")

    return HealthCheckResponse(status="ready")

