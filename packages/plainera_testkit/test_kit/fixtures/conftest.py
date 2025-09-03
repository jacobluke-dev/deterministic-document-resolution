import pytest
from alembic import command
from sqlalchemy.orm import sessionmaker
from alembic.config import Config

from src.public_api.core.settings import db_settings

from db_manager.connection import DBManager
from utils.utils import get_project_path


@pytest.fixture(scope="session")
def _session_factory(_engine):
    return sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)


# --- apply migrations ON THE SAME ENGINE -------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _apply_migrations_once(_engine):
    cfg = Config(get_project_path("alembic.ini", raise_error=True))
    with _engine.connect() as conn:
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, "head")
        # prove the version table exists in our schema
        conn.exec_driver_sql(f'SELECT 1 FROM "{db_settings.DB_SCHEMA}".alembic_version LIMIT 1')


# --- DBManager used everywhere (app + DI) ------------------------------------

class TestDBManager(DBManager):
    def __init__(self, engine, session_factory, allowed_tables):
        super().__init__(engine=engine, session_factory=session_factory, allowed_tables=allowed_tables)
