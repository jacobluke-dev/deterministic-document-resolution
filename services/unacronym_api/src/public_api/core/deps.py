from __future__ import annotations

from asyncio import Semaphore
from collections.abc import Iterator
from typing import Annotated, Any

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
    """
    Return the application-scoped DB manager from FastAPI app state.

    The DB manager is initialised during application startup in `public_api.main.lifespan()`
    and stored on `app.state.dbm`. This dependency is the single source of truth for DB
    access across the API layer, ensuring all request-scoped repositories/services share
    the same underlying DB wiring.

    In local/dev configurations where the database is disabled (e.g. `DATABASE_DISABLED=true`),
    `app.state.dbm` may be a no-op/null DB manager that safely returns no results for reads.

    Args:
        request: The current FastAPI request, used to access `request.app.state`.

    Returns:
        Any: The DB manager-like object stored at `request.app.state.dbm`.
    """
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
    """
    Return the request timeout in milliseconds for API operations.

    This is a lightweight dependency that exposes the current timeout value
    (from `app_settings`) for request-scoped services (e.g. `ResolveService`).
    Keeping this as a dependency makes it easy to override in tests without
    rebuilding the app container.

    Returns:
        int: Timeout in milliseconds (e.g. 5000).
    """
    return app_settings.REQUEST_TIMEOUT_MS


def get_glossary_repo(
    dbm: Annotated[Any, Depends(get_dbm)],
) -> GlossaryRepository:
    """
    Provide a request-scoped glossary repository.

    This wraps the app-scoped DB manager (`dbm`) in a small repository object
    used by service-layer code to perform read-only glossary lookups. The
    repository is constructed per request to make dependency overrides and
    test isolation straightforward.

    Args:
        dbm: DB manager retrieved from the FastAPI app state via `get_dbm()`.

    Returns:
        GlossaryRepository: Repository instance for glossary reads.
    """
    return GlossaryRepository(dbm=dbm)


def get_resolve_service(
    semaphore: Annotated[Semaphore | None, Depends(get_semaphore)],
    glossary_repo: Annotated[GlossaryRepository, Depends(get_glossary_repo)],
    timeout_ms: Annotated[int, Depends(get_request_timeout_ms)],
) -> ResolveService:
    """
    Provide a request-scoped ResolveService.

    Constructs the service using app-scoped collaborators (resolver, semaphore)
    and request-scoped collaborators (glossary repository, timeout value). The
    service is created per request so tests can override dependencies (e.g.
    resolver, semaphore, timeout) cleanly via `app.dependency_overrides`.

    Args:
        semaphore: Optional concurrency limiter injected via `get_semaphore()`.
        glossary_repo: Glossary repository injected via `get_glossary_repo()`.
        timeout_ms: Request timeout in milliseconds injected via `get_request_timeout_ms()`.

    Returns:
        ResolveService: Fully configured service instance for `/v1/resolve`.
    """
    return ResolveService(
        glossary_repo=glossary_repo,
        semaphore=semaphore,
        request_timeout_ms=timeout_ms,
    )
