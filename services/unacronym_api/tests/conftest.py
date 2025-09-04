# services/unacronym_api/tests/conftest.py
import os
import time

import pytest
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from public_api.core import deps
from public_api.core.settings import AppSettings, db_settings
from public_api.main import create_app
from test_kit.fixtures import TestDBManager
from utils.utils import get_project_path

# --- env ---------------------------------------------------------------------

IN_CI = os.getenv("CI") == "true" or os.getenv("GITLAB_CI") == "true"
os.environ.setdefault("ENVIRONMENT", "TEST")
if not IN_CI:
    os.environ.pop("DATABASE_URL", None)

# --- postgres boot ------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _db_ready(TEST_DB_URL):
   """ Only wait when using an external CI service URL"""
   if os.getenv("DATABASE_URL"):
        import psycopg
        dsn = os.environ["DATABASE_URL"].replace("+psycopg", "")
        for _ in range(120):
            try:
                with psycopg.connect(dsn, connect_timeout=2) as conn, conn.cursor() as cur:
                    cur.execute("SELECT 1")
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("DB not ready after 120s")


# --- apply migrations ON THE SAME ENGINE -------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _apply_migrations_once(engine_factory):
    cfg = Config(get_project_path("services/unacronym_api/alembic.ini", raise_error=True))
    with engine_factory.connect() as conn:
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, "head")
        # prove the version table exists in our schema
        conn.exec_driver_sql(f'SELECT 1 FROM "{db_settings.DB_SCHEMA}".alembic_version LIMIT 1')


# --- app client ---------------------------------------------------------------

@pytest.fixture()
async def client(engine_factory, session_factory, monkeypatch):
    # Ensure anything that reads env gets a valid DSN (not strictly required once we patch make_dbm)
    os.environ["DATABASE_URL"] = engine_factory.url.render_as_string(hide_password=False)

    # Force lifespan to use OUR DBM/engine (single engine everywhere)
    monkeypatch.setattr(
        "src.public_api.main.make_dbm",
        lambda test_mode=False: TestDBManager(
            engine=engine_factory,
            session_factory=session_factory,
            allowed_tables={"glossary_entries", "acronym_aliases"},
        ),
        raising=False,
    )

    app = create_app(settings=AppSettings(RUN_DB_MIGRATIONS=False, ENABLE_DOCS=False))

    # (Optional) also override dependency to be explicit
    app.dependency_overrides[deps.get_dbm] = lambda: TestDBManager(
        engine=engine_factory,
        session_factory=session_factory,
        allowed_tables={"glossary_entries", "acronym_aliases"},
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac
