# test_kit/fixtures.py
import os
import time
import urllib.parse
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from public_api.core.settings import db_settings
from document_resolution_core.db_manager.connection import DBManager
from document_resolution_core.utils.utils import get_project_path

def _normalize(url: str) -> str:
    return url.replace("postgresql+psycopg2://", "postgresql+psycopg://")

def _install_search_path(engine, schema: str):
    @event.listens_for(engine, "connect")
    def _set_search_path(dbapi_conn, _):
        with dbapi_conn.cursor() as cur:
            cur.execute(f'SET search_path TO "{schema}", public')

@pytest.fixture(scope="session")
def TEST_DB_URL():
    """Choose DB for tests:
    1) TEST_DB_URL env (host-friendly) → use it
    2) CI && DATABASE_URL set → use it
    3) else start a testcontainers Postgres
    """
    tc = None
    try:
        if os.getenv("TEST_DB_URL"):
            url = _normalize(os.environ["TEST_DB_URL"])
        elif os.getenv("CI") == "true" and os.getenv("DATABASE_URL"):
            url = _normalize(os.environ["DATABASE_URL"])
        else:
            from testcontainers.postgres import PostgresContainer
            tc = PostgresContainer("postgres:15-alpine")
            tc.start()
            url = _normalize(tc.get_connection_url())
        yield url
    finally:
        if tc:
            tc.stop()

@pytest.fixture(scope="session")
def db_ready(TEST_DB_URL):
    # If we’re using a remote/host DSN, optionally wait a bit
    parsed = urllib.parse.urlparse(TEST_DB_URL)
    host = parsed.hostname or ""
    if host not in {"localhost", "127.0.0.1"}:
        return  # container branch or CI will be handled elsewhere
    import psycopg
    dsn = TEST_DB_URL.replace("+psycopg", "")
    for _ in range(int(os.getenv("DOCUMENT_RESOLUTION_TESTKIT_DB_WAIT_SECONDS", "10"))):
        try:
            with psycopg.connect(dsn, connect_timeout=2) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            break
        except Exception:
            time.sleep(1)

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
    cfg = Config(get_project_path(os.getenv("ALEMBIC_INI_PATH", "services/unacronym_api/alembic.ini"), raise_error=True))
    with engine_factory.connect() as conn:
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, "head")
        conn.exec_driver_sql(f'SELECT 1 FROM "{db_settings.DB_SCHEMA}".alembic_version LIMIT 1')

class TestDBManager(DBManager):
    def __init__(self, engine, session_factory, allowed_tables):
        super().__init__(engine=engine, session_factory=session_factory, allowed_tables=allowed_tables)

@pytest.fixture(name="dbm")
def dbm(apply_migrations_once, engine_factory, session_factory):
    return TestDBManager(
        engine_factory,
        session_factory,
        {"glossary_acronyms", "glossary_meanings", "glossary_variants"},
    )
