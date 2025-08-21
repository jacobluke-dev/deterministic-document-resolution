from __future__ import annotations

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from public_api.api.routers.health import router as health_router
from public_api.core.logging import configure_logging
from public_api.core.middleware import (
    AccessLogMiddleware,
    BodySizeLimitMiddleware,
    RequestIDMiddleware,
)
from public_api.core.settings import AppSettings

__version__ = "0.1.0"


def create_app(settings: AppSettings | None = None) -> FastAPI:
    settings = settings or AppSettings()

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

    # Middleware order: size limit → request-id → access log → CORS
    app.add_middleware(BodySizeLimitMiddleware, max_body_bytes=settings.MAX_BODY_BYTES)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins or [],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    return app
