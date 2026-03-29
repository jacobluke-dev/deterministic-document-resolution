
import os

# --- only now import modules that may construct settings/engines ---
import time

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from plainera_core.utils.utils import find_project_root, get_project_path
from public_api.core.di import deps
from public_api.core.settings import AppSettings, db_settings
from public_api.main import create_app
from test_kit.fixtures import TestDBManager

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("AUTH_DISABLED", "false")
os.environ.setdefault("ENVIRONMENT", "TEST")

@pytest.fixture
def anyio_backend():
    # Force AnyIO tests to run on asyncio, we don't need Trio installed.
    return "asyncio"

# --- env ---------------------------------------------------------------------

ROOT = find_project_root(__file__, markers=(".git", "pyproject.toml"))
ENV_PATH = ROOT / ".env"

ALLOWED_TABLES = {
    "glossary_acronym",
    "glossary_meaning",
    "glossary_variant",
    "api_keys",
    "api_usage_daily",
    "api_usage_minute",
}

try:
    from dotenv import load_dotenv
    load_dotenv(ENV_PATH)  # loads the project-root .env
except Exception:
    pass

os.environ.setdefault("ENVIRONMENT", "TEST")


IN_CI = os.getenv("CI") == "true" or os.getenv("GITLAB_CI") == "true"
os.environ.setdefault("ENVIRONMENT", "TEST")
if not IN_CI:
    os.getenv("DATABASE_URL", None)

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


@pytest_asyncio.fixture
async def client(engine_factory, session_factory, monkeypatch):
    from public_api.core.auth.api_keys import generate_key, hash_secret
    from sqlalchemy import text
    key_id, secret, full = generate_key("test")
    key_hash = hash_secret(secret, scheme="argon2id")

    with session_factory() as s:
        s.execute(
            text(
                """
                INSERT INTO unacronym.api_keys (key_id, key_hash, prefix, scopes, is_active, created_at)
                VALUES (:key_id, :key_hash, :prefix, '{}'::text[], true, now())
                """
            ),
            {"key_id": key_id, "key_hash": key_hash, "prefix": "test"},
        )
        s.commit()
    # Ensure anything that reads env gets a valid DSN (not strictly required once we patch make_dbm)
    os.environ["DATABASE_URL"] = engine_factory.url.render_as_string(hide_password=False)

    # Force lifespan to use OUR DBM/engine (single engine everywhere)
    monkeypatch.setattr(
        "public_api.main.make_dbm",
        lambda test_mode=False: TestDBManager(
            engine=engine_factory,
            session_factory=session_factory,
            allowed_tables=ALLOWED_TABLES
        ),
        raising=False,
    )

    app = create_app(settings=AppSettings(RUN_DB_MIGRATIONS=False, ENABLE_DOCS=False))

    # (Optional) also override dependency to be explicit
    app.dependency_overrides[deps.get_dbm] = lambda: TestDBManager(
        engine=engine_factory,
        session_factory=session_factory,
        allowed_tables=ALLOWED_TABLES,
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
        headers={"X-API-Key": full},
    ) as ac:
        yield ac


@pytest_asyncio.fixture
async def client_no_auth(engine_factory, session_factory, monkeypatch):
    os.environ["DATABASE_URL"] = engine_factory.url.render_as_string(hide_password=False)

    monkeypatch.setattr(
        "public_api.main.make_dbm",
        lambda test_mode=False: TestDBManager(
            engine=engine_factory,
            session_factory=session_factory,
            allowed_tables=ALLOWED_TABLES,
        ),
        raising=False,
    )

    app = create_app(settings=AppSettings(RUN_DB_MIGRATIONS=False, ENABLE_DOCS=False))

    app.dependency_overrides[deps.get_dbm] = lambda: TestDBManager(
        engine=engine_factory,
        session_factory=session_factory,
        allowed_tables=ALLOWED_TABLES,
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


@pytest.fixture
def _patch(monkeypatch):
    """Return a helper that patches names in a function's global namespace.

    The returned helper replaces entries in ``func.__globals__`` using pytest's
    ``monkeypatch.setitem``. This is useful when code under test resolves
    imported symbols from the module where the function is defined, rather than
    from the original import source. It is particularly handy for patching
    module-level collaborators such as loggers, sinks, helper functions, or
    imported dependencies exactly where they are used.

    Args:
        monkeypatch: Built-in pytest fixture used to apply reversible test-time
            patches.

    Returns:
        A callable that accepts a target function plus keyword replacements, where
        each keyword is the global name to replace and each value is the test
        implementation to inject.

    Example:
        Patch ``message_logger`` in the globals of ``detect`` so the wrapped
        method does not write to a real sink during tests::

            _patch(
                detector.detect.__func__,
                message_logger=lambda *args, **kwargs: None,
            )
    """
    def _apply(func, **replacements):
        g = func.__globals__
        for name, impl in replacements.items():
            monkeypatch.setitem(g, name, impl)
        return func
    return _apply
