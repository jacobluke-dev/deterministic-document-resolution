from __future__ import annotations

from typing import Protocol, cast

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from db_manager.factory import make_dbm
from observability.http.request_id import RequestIDMiddleware
from observability.observability.access_middleware import access_middleware
from observability.http.body_limit import BodySizeLimitMiddleware


from starlette.datastructures import State
from starlette.middleware.cors import CORSMiddleware

from src.public_api.api.routers.errors import map_length_validation_to_413
from src.public_api.api.routers.health import router as health_router
from src.public_api.api.routers.resolve import router as resolve_router
from src.public_api.core.logging import configure_logging
from src.public_api.core.settings import AppSettings, app_settings

__version__ = "0.1.0"


class HasState(Protocol):
    state: State

def create_app(settings: AppSettings  = app_settings) -> FastAPI:
    settings = settings

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
    )

    cast(HasState, app).state.dbm = make_dbm()

    # Middleware order: size limit → request-id → access log → CORS
    app.add_middleware(BodySizeLimitMiddleware, max_body_bytes=settings.MAX_BODY_BYTES)
    app.add_middleware(RequestIDMiddleware)
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
    access_middleware(app)

    return app
