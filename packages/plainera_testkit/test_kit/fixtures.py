import os, time
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from src.public_api.core.settings import db_settings
from db_manager.connection import DBManager
from utils.utils import get_project_path

def _normalize(url: str) -> str:
    return url.replace("postgresql+psycopg2://", "postgresql+psycopg://")

@pytest.fixture(scope="session", name="TEST_DB_URL")
def _test_db_url():
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        yield _normalize(env_url)
        return
    from testcontainers.postgres import PostgresContainer
    with PostgresContainer("postgres:15-alpine") as pg:
        yield _normalize(pg.get_connection_url())

def _install_search_path(engine, schema: str):
    @event.listens_for(engine, "connect")
    def _set_search_path(dbapi_conn, _):
        with dbapi_conn.cursor() as cur:
            cur.execute(f'SET search_path TO "{schema}", public')
# test_kit/fixtures.py

@pytest.fixture(scope="session", name="db_ready")
def _db_ready(TEST_DB_URL):
    if os.getenv("DATABASE_URL") and os.getenv("CI") == "true":
        import psycopg, time
        dsn = os.environ["DATABASE_URL"].replace("+psycopg", "")
        WAIT_SECS = int(os.getenv("PLAINERA_TESTKIT_DB_WAIT_SECONDS", "5"))
        for _ in range(WAIT_SECS):
            try:
                with psycopg.connect(dsn, connect_timeout=2) as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError(f"DB not ready after {WAIT_SECS}s")

@pytest.fixture(scope="session", name="engine_factory")
def engine_factory(TEST_DB_URL, db_ready):
    engine = create_engine(TEST_DB_URL, future=True, pool_pre_ping=True)
    _install_search_path(engine, db_settings.DB_SCHEMA)
    with engine.connect() as c:
        sp = c.exec_driver_sql("show search_path").scalar()
        assert db_settings.DB_SCHEMA in sp, f"search_path={sp}"
    try:
        yield engine
    finally:
        engine.dispose()

@pytest.fixture(scope="session", name="session_factory")
def session_factory(engine_factory):
    return sessionmaker(bind=engine_factory, autoflush=False, autocommit=False, future=True)

@pytest.fixture(scope="session", name="apply_migrations_once")
def apply_migrations_once(engine_factory):
    ini_path = os.getenv("ALEMBIC_INI_PATH", "alembic.ini")
    cfg = Config(get_project_path(ini_path, raise_error=True))
    with engine_factory.connect() as conn:
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, "head")
        conn.exec_driver_sql(f'SELECT 1 FROM "{db_settings.DB_SCHEMA}".alembic_version LIMIT 1')

class TestDBManager(DBManager):
    def __init__(self, engine, session_factory, allowed_tables):
        super().__init__(engine=engine, session_factory=session_factory, allowed_tables=allowed_tables)

@pytest.fixture(name="dbm")
def dbm(apply_migrations_once, engine_factory, session_factory):
    return TestDBManager(engine_factory, session_factory, {"glossary_entries", "acronym_aliases"})
