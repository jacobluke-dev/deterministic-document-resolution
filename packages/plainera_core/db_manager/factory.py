from typing import Optional

from db_manager.connection import DBManager
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from src.public_api.core.settings import db_settings


def make_dbm(url: Optional[str] = None) -> DBManager:
    """Factory for a DBManager tied to the current db_settings.

    Args:
        url (Optional[str]): The database URL.
    Returns:
        DBManager: DBManager instance
    """
    dsn = url or db_settings.database_url
    engine = create_engine(
        dsn,
        pool_pre_ping=True,
    )
    SessionLocal: sessionmaker[Session] = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )
    return DBManager(engine=engine, session_factory=SessionLocal)
