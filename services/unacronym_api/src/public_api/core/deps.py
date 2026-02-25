from __future__ import annotations

from asyncio import Semaphore
from collections.abc import Iterator
from typing import Any

from fastapi import Depends, Request
from plainera_core.db_manager.connection import DBManager
from sqlalchemy.orm import Session

from public_api.core.factory import create_resolver
from public_api.core.providers import AcronymResolverLike
from public_api.core.services.resolve_service import ResolveService
from public_api.core.settings import app_settings
from public_api.db.repos.glossary_repo import GlossaryRepository


class AppContainer:
    """Application-scoped dependency container.

    This container is created once at startup and holds shared,
    long-lived resources for the FastAPI application such as
    the resolver and concurrency-limiting semaphore.

    Attributes:
        resolver: The acronym resolver instance created by
            `create_resolver()`.
        semaphore (Semaphore | None): An asyncio semaphore
            that limits the number of concurrent in-flight
            requests. set only if `MAX_INFLIGHT` is greater
            than zero in app settings.
    """

    def __init__(self) -> None:
        """Initialize the application container.

        Creates the resolver and configures a semaphore if concurrency
        limiting is enabled via settings.
        """
        self.resolver = create_resolver()
        self.semaphore: Semaphore | None = None
        if app_settings.MAX_INFLIGHT and app_settings.MAX_INFLIGHT > 0:
            self.semaphore = Semaphore(app_settings.MAX_INFLIGHT)


container = AppContainer()


def get_resolver() -> AcronymResolverLike:
    """Provide the global resolver instance.

    Used as a FastAPI dependency to inject the application’s
    resolver into request handlers.

    Returns:
        The singleton resolver created at startup.
    """
    return container.resolver


def get_semaphore() -> Semaphore | None:
    """Provide the global concurrency semaphore.

    Used as a FastAPI dependency to throttle concurrent requests
    if `MAX_INFLIGHT` > 0. Returns None if concurrency limiting
    is disabled.

    Returns:
        Semaphore | None: The global semaphore if concurrency limiting
        is enabled, otherwise None.
    """
    return container.semaphore


def get_dbm(request: Request) -> Any:
    # Single source of truth: set in main.lifespan()
    return request.app.state.dbm


def get_session(dbm: DBManager) -> Iterator[Session]:
    """Yield a transactional SQLAlchemy Session.

    This wraps `DBManager.session()` so routes can depend on a ready-to-use
    session with commit/rollback/close handled automatically.

    Yields:
        Iterator[Session]: An active Session for the request scope.
    """
    with dbm.session() as s:
        yield s


def get_request_timeout_ms() -> int:
    return app_settings.REQUEST_TIMEOUT_MS


def get_glossary_repo(dbm: Any = Depends(get_dbm)) -> GlossaryRepository:
    return GlossaryRepository(dbm=dbm)


def get_resolve_service(
    resolver: AcronymResolverLike = Depends(get_resolver),
    semaphore: Semaphore | None = Depends(get_semaphore),
    glossary_repo: GlossaryRepository = Depends(get_glossary_repo),
    timeout_ms: int = Depends(get_request_timeout_ms),
) -> ResolveService:
    # built per-request so test-time overrides (timeout/resolver) apply cleanly
    return ResolveService(
        resolver=resolver,
        glossary_repo=glossary_repo,
        semaphore=semaphore,
        request_timeout_ms=timeout_ms,
    )
