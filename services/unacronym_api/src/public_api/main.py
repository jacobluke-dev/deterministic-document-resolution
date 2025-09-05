from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from observability.http.body_limit import BodySizeLimitMiddleware
from observability.http.request_id import RequestIDMiddleware
from observability.observability.access_middleware import access_middleware
from plainera_core.db_manager.factory import make_dbm
from sqlalchemy.engine import Engine
from starlette.middleware.cors import CORSMiddleware

from public_api.api.routers.errors import map_length_validation_to_413
from public_api.api.routers.health import router as health_router
from public_api.api.routers.resolve import router as resolve_router
from public_api.core.logging import configure_logging
from public_api.core.settings import AppSettings, app_settings

__version__ = "0.1.0"

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, Any]:
    # init
    state = app.state  # noqa: ignore[assignment]
    dbm = make_dbm(test_mode=False)
    state.dbm = dbm  # noqa: ignore[index]  # AppState["dbm"]
    try:
        yield
    finally:
        # dispose cleanly
        engine: Engine = dbm.engine  # noqa: ignore[attr-defined]
        engine.dispose()

def create_app(settings: AppSettings | None = None) -> FastAPI:
    settings = settings or app_settings
    configure_logging(settings.LOG_LEVEL)

    docs_url = "/docs" if settings.ENABLE_DOCS else None
    redoc_url = "/redoc" if settings.ENABLE_DOCS else None
    openapi_url = "/openapi.json" if settings.ENABLE_DOCS else None

    app = FastAPI(
        title="Unacronym API",
        version=__version__,
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
        lifespan=lifespan,
    )

    # Middleware order: size limit → request-id → access log → CORS
    app.add_middleware(BodySizeLimitMiddleware, max_body_bytes=settings.MAX_BODY_BYTES)
    app.add_middleware(RequestIDMiddleware)
    access_middleware(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins or [],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(resolve_router)
    app.add_exception_handler(RequestValidationError, map_length_validation_to_413)
    return app
