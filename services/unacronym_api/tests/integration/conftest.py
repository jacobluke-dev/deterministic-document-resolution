from __future__ import annotations

import os
import time

import psycopg
import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from plainera_core.utils.utils import get_project_path
from public_api.core.di import deps
from public_api.core.auth.api_keys import generate_key, hash_secret
from public_api.core.settings import AppSettings, db_settings
from public_api.db.models import GlossaryAcronym, GlossaryMeaning
from public_api.main import create_app
from sqlalchemy import text
from sqlalchemy.inspection import inspect
from test_kit.fixtures import TestDBManager

pytestmark = [pytest.mark.integration]

ALLOWED_TABLES = {
    "glossary_acronyms",
    "glossary_meanings",
    "glossary_variants",
    "api_keys",
    "api_usage_daily",
    "api_usage_minute",
}

TABLES_TO_CLEAN = (
    "glossary_acronyms",
    "glossary_meanings",
    "glossary_variants",
    "api_keys",
    "api_usage_daily",
    "api_usage_minute",
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session", autouse=True)
def _db_ready() -> None:
    dsn = os.environ["DATABASE_URL"].replace("+psycopg", "")
    for _ in range(120):
        try:
            with psycopg.connect(dsn, connect_timeout=2) as conn, conn.cursor() as cur:
                cur.execute("SELECT 1")
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("DB not ready after 120s")


@pytest.fixture(scope="session", autouse=True)
def _apply_migrations_once(engine_factory) -> None:
    cfg = Config(get_project_path("services/unacronym_api/alembic.ini", raise_error=True))
    with engine_factory.connect() as conn:
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, "head")
        conn.exec_driver_sql(
            f'SELECT 1 FROM "{db_settings.DB_SCHEMA}".alembic_version LIMIT 1'
        )


@pytest.fixture(autouse=True)
def _clean_tables(engine_factory, session_factory) -> None:
    inspector = inspect(engine_factory)
    existing_tables = set(inspector.get_table_names(schema=db_settings.DB_SCHEMA))
    tables = [table for table in TABLES_TO_CLEAN if table in existing_tables]

    if not tables:
        raise RuntimeError(
            f"No expected tables found in schema {db_settings.DB_SCHEMA!r}. "
            f"Existing tables: {sorted(existing_tables)!r}"
        )

    with session_factory() as s:
        for table in tables:
            s.execute(
                text(
                    f'TRUNCATE TABLE "{db_settings.DB_SCHEMA}"."{table}" '
                    "RESTART IDENTITY CASCADE"
                )
            )
        s.commit()

@pytest.fixture
def app(engine_factory, session_factory, monkeypatch):
    os.environ["DATABASE_URL"] = engine_factory.url.render_as_string(hide_password=False)

    dbm = TestDBManager(
        engine=engine_factory,
        session_factory=session_factory,
        allowed_tables=set(ALLOWED_TABLES),
    )

    monkeypatch.setattr(
        "public_api.main.make_dbm",
        lambda test_mode=False: dbm,
        raising=False,
    )

    app = create_app(
        settings=AppSettings(
            RUN_DB_MIGRATIONS=False,
            ENABLE_DOCS=False,
        )
    )
    app.dependency_overrides[deps.get_dbm] = lambda: dbm
    return app


@pytest_asyncio.fixture
async def client(app, session_factory):
    key_id, secret, full = generate_key("test")
    key_hash = hash_secret(secret, scheme="argon2id")

    with session_factory() as s:
        s.execute(
            text(
                f"""
                INSERT INTO "{db_settings.DB_SCHEMA}"."api_keys"
                    (key_id, key_hash, prefix, scopes, is_active, created_at)
                VALUES
                    (:key_id, :key_hash, :prefix, '{{}}'::text[], true, now())
                """
            ),
            {
                "key_id": key_id,
                "key_hash": key_hash,
                "prefix": "test",
            },
        )
        s.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
        headers={"X-API-Key": full},
    ) as ac:
        yield ac


@pytest_asyncio.fixture
async def client_no_auth(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as ac:
        yield ac


@pytest.fixture
def seed_glossary(session_factory):
    with session_factory() as s:
        def upsert(acronym: str, definition: str, *, domain: str = "general") -> None:
            normalized = acronym.lower()

            ga = (
                s.query(GlossaryAcronym)
                .filter(GlossaryAcronym.tenant_id.is_(None))
                .filter(GlossaryAcronym.normalized == normalized)
                .first()
            )
            if ga is None:
                ga = GlossaryAcronym(
                    tenant_id=None,
                    acronym=acronym,
                    normalized=normalized,
                    is_active=True,
                )
                s.add(ga)
                s.flush()
            else:
                ga.acronym = acronym
                ga.is_active = True

            gm = (
                s.query(GlossaryMeaning)
                .filter(GlossaryMeaning.acronym_id == ga.id)
                .filter(GlossaryMeaning.domain == domain)
                .first()
            )
            if gm is None:
                s.add(
                    GlossaryMeaning(
                        acronym_id=ga.id,
                        definition=definition,
                        domain=domain,
                        provenance="test",
                        is_active=True,
                    )
                )
            else:
                gm.definition = definition
                gm.provenance = "test"
                gm.is_active = True

        upsert("MPS", "Metropolitan Police Service.")
        upsert("GPU", "Graphics Processing Unit.")
        s.commit()
