from __future__ import annotations

from asyncio import Semaphore
from collections.abc import Iterator

from sqlalchemy.orm import Session

from src.public_api.core.providers import AcronymResolverLike, create_resolver
from src.public_api.core.settings import app_settings
from src.public_api.core.settings import db_settings as db_settings
from src.public_api.db.db_manager.connection import DBManager


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
            requests. Set only if `MAX_INFLIGHT` is greater
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

        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        engine = create_engine(db_settings.database_url, pool_pre_ping=True)
        SessionLocal: sessionmaker[Session] = sessionmaker(
            bind=engine, autoflush=False, autocommit=False
        )
        self.dbm = DBManager(engine=engine, session_factory=SessionLocal)


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


def get_dbm() -> DBManager:
    """Return the app-scoped DBManager instance.

    DBManager: The DB manager attached to `app.state.dbm`.
    """
    return container.dbm


def get_session(dbm: DBManager) -> Iterator[Session]:
    """Yield a transactional SQLAlchemy Session.

    This wraps `DBManager.session()` so routes can depend on a ready-to-use
    session with commit/rollback/close handled automatically.

    Yields:
        Iterator[Session]: An active Session for the request scope.
    """
    with dbm.session() as s:
        yield s
