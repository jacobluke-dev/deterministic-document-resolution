from typing import Optional

from db_manager.connection import DBManager
from sqlalchemy import NullPool, create_engine
from sqlalchemy.orm import Session, sessionmaker
from src.public_api.core.settings import db_settings


def make_dbm(url: Optional[str] = None, *, test_mode: bool = False) -> DBManager:
    """Create a DBManager bound to the configured database URL.

    Args:
        url (Optional[str]): Override database URL. If None, use db_settings.database_url.
        test_mode (bool): If True, use a NullPool so connections are not reused
            across tests. Recommended for unit/feature tests to avoid cross-test leakage.

    Returns:
        DBManager: A new DBManager instance with its own SQLAlchemy engine
        and session factory.
    """
    dsn = url or db_settings.database_url
    engine = create_engine(
        dsn,
        pool_pre_ping=True,
        future=True,
        poolclass=NullPool if test_mode else None,
    )
    SessionLocal: sessionmaker[Session] = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        future=True,
    )
    return DBManager(engine=engine, session_factory=SessionLocal)
