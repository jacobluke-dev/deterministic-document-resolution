from __future__ import annotations

from contextlib import AbstractContextManager, asynccontextmanager
from types import TracebackType
from typing import Any, AsyncGenerator, Literal

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from observability.http.body_limit import BodySizeLimitMiddleware
from observability.http.request_id import RequestIDMiddleware
from observability.logger.access_middleware import access_middleware
from document_resolution_core.db_manager.factory import make_dbm
from document_resolution.nlp.extraction.tiers.semantic import _load_st_model
from sqlalchemy.engine import Engine
from starlette.exceptions import HTTPException
from starlette.middleware.cors import CORSMiddleware

from public_api.api.routers.exception_handlers import map_http_exception, map_length_validation_to_413
from public_api.api.routers.health import router as health_router
from public_api.api.routers.resolve import router as resolve_router
from public_api.core.logging import configure_logging
from public_api.core.services.api_abuse_protection import (
    QuotaExceededError,
    RateLimitExceededError,
    quota_exceeded_handler,
    rate_limited_handler,
)
from public_api.core.settings import AppSettings, app_settings, db_settings

__version__ = "0.1.0"


class _NullDBSession(AbstractContextManager[Any]):
    """A context manager that fails on entry (used when DB is disabled)."""

    def __enter__(self) -> Any:
        raise RuntimeError("Database is disabled (DATABASE_DISABLED=true).")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        return False


class _NullDBManager:
    """DBManager stand-in used when DATABASE_DISABLED=true.

    Keeps dependency signatures stable while making DB-backed features no-op.
    """

    def select_one_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any] | None:
        return None

    def session(self) -> AbstractContextManager[Any]:
        return _NullDBSession()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, Any]:
    state = app.state

    # ---- Tier-2 model (startup) ----
    state = app.state

    state.tier2_model = None

    if app_settings.TIER2_ENABLED:
        try:
            state.tier2_model = _load_st_model(
                app_settings.TIER2_MODEL_NAME,
                cache_folder=app_settings.HF_CACHE_DIR,
            )
        except ModuleNotFoundError:
            # sentence_transformers / torch not installed
            if app_settings.TIER2_STRICT:
                raise
            state.tier2_model = None

    # ---- DBM (startup) ----
    if db_settings.DATABASE_DISABLED:
        state.dbm = _NullDBManager()
        yield
        return

    dbm = make_dbm(test_mode=False)
    state.dbm = dbm
    try:
        yield
    finally:
        engine: Engine = dbm.engine
        engine.dispose()


def create_app(settings: AppSettings | None = None) -> FastAPI:
    settings = settings or app_settings
    configure_logging(settings.LOG_LEVEL)

    docs_url = "/docs" if settings.ENABLE_DOCS else None
    redoc_url = "/redoc" if settings.ENABLE_DOCS else None
    openapi_url = "/openapi.json" if settings.ENABLE_DOCS else None

    app = FastAPI(
        title="document_resolution API",
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
    app.add_exception_handler(HTTPException, map_http_exception)
    app.add_exception_handler(QuotaExceededError, quota_exceeded_handler)
    app.add_exception_handler(RateLimitExceededError, rate_limited_handler)
    app.state.settings = settings

    return app
