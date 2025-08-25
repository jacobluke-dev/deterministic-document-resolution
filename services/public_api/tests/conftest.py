# tests/conftest.py

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from public_api.core import deps
from public_api.db.db_manager.connection import DBManager

# ⬇️ import your Base + models here
from public_api.db.models import Base  # provides Base.metadata
from public_api.main import create_app

# If you need specific tables for seeding:
# from public_api.db.models import GlossaryEntry, AcronymAlias

@pytest.fixture(scope="session")
def _engine():
    # one process-wide in-memory DB shared across threads/connections
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    # 1) Create schema
    Base.metadata.create_all(engine)

    # 2) Create a dummy alembic_version so any migration check is happy
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE IF NOT EXISTS alembic_version (version_num VARCHAR PRIMARY KEY)"
        )
        conn.exec_driver_sql(
            "INSERT OR IGNORE INTO alembic_version (version_num) VALUES ('test')"
        )

    return engine


@pytest.fixture(scope="session")
def _session_factory(_engine):
    return sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)


@pytest.fixture(scope="session", autouse=True)
def _seed_minimal_data(_engine, _session_factory):
    # Seed the absolute minimum your endpoint relies on.
    # If resolve() is pure and doesn’t hit DB, this is harmless; if it
    # looks up acronyms/aliases, add a couple of rows to make tests deterministic.
    # Example (adapt field names to your actual schema):
    # with _engine.begin() as conn:
    #     pass

    # Example with ORM session:
    # with _session_factory() as s:
    #     s.add_all([
    #         GlossaryEntry(term="Metropolitan Police Service", acronym="MPS"),
    #         GlossaryEntry(term="Alpha Beta Charlie", acronym="ABC"),
    #     ])
    #     s.commit()
    yield


class TestDBManager(DBManager):
    """
    Thin adapter so your code can use the same API in tests.
    """

    def __init__(self, engine, session_factory, allowed_tables):
        super().__init__(engine=engine, session_factory=session_factory, allowed_tables=allowed_tables)

@pytest.fixture()
def dbm(_engine, _session_factory):
    return TestDBManager(
        engine=_engine,
        session_factory=_session_factory,
        allowed_tables={"glossary_entries"},
    )


@pytest.fixture()
async def client(_engine, _session_factory, monkeypatch):
    app = create_app()

    # Optional: if your startup tries to run migrations, turn that off in tests
    # (safe no-op if the setting doesn’t exist)
    monkeypatch.setattr(
        "public_api.core.settings.app_settings.RUN_DB_MIGRATIONS", False, raising=False
    )

    test_dbm = TestDBManager(
        engine=_engine,
        session_factory=_session_factory,
        allowed_tables={"glossary_entries", "acronym_aliases"},
    )
    app.dependency_overrides[deps.get_dbm] = lambda: test_dbm

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac
