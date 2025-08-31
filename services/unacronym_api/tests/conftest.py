import os
import time

import pytest
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient

from db_manager.connection import DBManager
from src.public_api.core import deps
from src.public_api.main import create_app
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

IN_CI = os.getenv("CI") == "true" or os.getenv("GITLAB_CI") == "true"
os.environ.setdefault("ENVIRONMENT", "TEST")
if not IN_CI:
    os.environ.pop("DATABASE_URL", None)



def _normalize(url: str) -> str:
    return url.replace("postgresql+psycopg2://", "postgresql+psycopg://")

@pytest.fixture(scope="session")
def TEST_DB_URL():
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        yield _normalize(env_url)
        return

    # Local fallback: testcontainers (requires Docker)
    from testcontainers.postgres import PostgresContainer
    with PostgresContainer("postgres:15-alpine") as pg:
        yield _normalize(pg.get_connection_url())

@pytest.fixture(scope="session", autouse=True)
def _db_ready(TEST_DB_URL):
    # Only wait when using CI service URL
    if os.getenv("DATABASE_URL"):
        import psycopg
        dsn = os.environ["DATABASE_URL"].replace("+psycopg", "")
        for _i in range(120):  # give runners more time
            try:
                with psycopg.connect(dsn, connect_timeout=2) as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("DB not ready after 120s")

@pytest.fixture(scope="session")
def _engine(TEST_DB_URL, _db_ready):
    engine = create_engine(TEST_DB_URL, future=True, pool_pre_ping=True)
    try:
        yield engine
    finally:
        engine.dispose()

@pytest.fixture(scope="session")
def _session_factory(_engine):
    return sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)

@pytest.fixture(scope="session", autouse=True)
def _apply_migrations_once(_engine):
    from src.public_api.utils.utils import get_project_path  # import after ENV set
    cfg = Config(get_project_path("alembic.ini", raise_error=True))
    with _engine.connect() as conn:
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, "head")


# ---- client fixture (avoid masked password!) -------------------

class TestDBManager(DBManager):
    def __init__(self, engine, session_factory, allowed_tables):
        super().__init__(engine=engine, session_factory=session_factory, allowed_tables=allowed_tables)

@pytest.fixture()
def dbm(_engine, _session_factory):
    return TestDBManager(_engine, _session_factory, {"glossary_entries", "acronym_aliases"})

@pytest.fixture()
async def client(_engine, _session_factory, monkeypatch):
    # Use a proper, unmasked URL string if you want the app to read env:
    os.environ["DATABASE_URL"] = _engine.url.render_as_string(hide_password=False)

    app = create_app()
    monkeypatch.setattr("public_api.core.settings.app_settings.RUN_DB_MIGRATIONS", False, raising=False)
    app.dependency_overrides[deps.get_dbm] = lambda: TestDBManager(
        engine=_engine,
        session_factory=_session_factory,
        allowed_tables={"glossary_entries", "acronym_aliases"},
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac
