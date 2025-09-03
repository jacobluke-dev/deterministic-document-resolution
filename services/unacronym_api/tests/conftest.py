# services/unacronym_api/tests/conftest.py
import os
import time
import pytest

from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient

from sqlalchemy import create_engine, event
from src.public_api.core.settings import db_settings, AppSettings
from src.public_api.utils.utils import get_project_path
from src.public_api.main import create_app
from src.public_api.core import deps
from test_kit.fixtures.conftest import _session_factory, TestDBManager


# --- env ---------------------------------------------------------------------

IN_CI = os.getenv("CI") == "true" or os.getenv("GITLAB_CI") == "true"
os.environ.setdefault("ENVIRONMENT", "TEST")
if not IN_CI:
    os.environ.pop("DATABASE_URL", None)


def _normalize(url: str) -> str:
    return url.replace("postgresql+psycopg2://", "postgresql+psycopg://")


# --- postgres boot ------------------------------------------------------------

@pytest.fixture(scope="session")
def TEST_DB_URL():
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        yield _normalize(env_url)
        return

    from testcontainers.postgres import PostgresContainer
    with PostgresContainer("postgres:15-alpine") as pg:
        yield _normalize(pg.get_connection_url())


@pytest.fixture(scope="session", autouse=True)
def _db_ready(TEST_DB_URL):
    # Only wait when using an external CI service URL
    if os.getenv("DATABASE_URL"):
        import psycopg
        dsn = os.environ["DATABASE_URL"].replace("+psycopg", "")
        for _ in range(120):
            try:
                with psycopg.connect(dsn, connect_timeout=2) as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("DB not ready after 120s")


def _install_search_path(engine, schema: str):
    @event.listens_for(engine, "connect")
    def _set_search_path(dbapi_conn, _):
        # psycopg3
        with dbapi_conn.cursor() as cur:
            cur.execute(f'SET search_path TO "{schema}", public')


@pytest.fixture(scope="session")
def _engine(TEST_DB_URL, _db_ready):
    engine = create_engine(
        TEST_DB_URL,
        future=True,
        pool_pre_ping=True,
    )
    _install_search_path(engine, db_settings.DB_SCHEMA)

    # sanity check
    with engine.connect() as c:
        sp = c.exec_driver_sql("show search_path").scalar()
        assert db_settings.DB_SCHEMA in sp, f"search_path={sp}"

    try:
        yield engine
    finally:
        engine.dispose()

# --- apply migrations ON THE SAME ENGINE -------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _apply_migrations_once(_engine):
    cfg = Config(get_project_path("alembic.ini", raise_error=True))
    with _engine.connect() as conn:
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, "head")
        # prove the version table exists in our schema
        conn.exec_driver_sql(f'SELECT 1 FROM "{db_settings.DB_SCHEMA}".alembic_version LIMIT 1')


# --- app client ---------------------------------------------------------------

@pytest.fixture()
async def client(_engine, _session_factory, monkeypatch):
    # Ensure anything that reads env gets a valid DSN (not strictly required once we patch make_dbm)
    os.environ["DATABASE_URL"] = _engine.url.render_as_string(hide_password=False)

    # Force lifespan to use OUR DBM/engine (single engine everywhere)
    monkeypatch.setattr(
        "src.public_api.main.make_dbm",
        lambda test_mode=False: TestDBManager(
            engine=_engine,
            session_factory=_session_factory,
            allowed_tables={"glossary_entries", "acronym_aliases"},
        ),
        raising=False,
    )

    app = create_app(settings=AppSettings(RUN_DB_MIGRATIONS=False, ENABLE_DOCS=False))

    # (Optional) also override dependency to be explicit
    app.dependency_overrides[deps.get_dbm] = lambda: TestDBManager(
        engine=_engine,
        session_factory=_session_factory,
        allowed_tables={"glossary_entries", "acronym_aliases"},
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac
