import os

from dotenv import load_dotenv
from plainera_core.db_manager.sessions import make_async_sessionmaker
from plainera_core.db_manager.sink_factory import make_universal_sink

load_dotenv()


def _to_asyncpg(url: str) -> str:
    # handle postgresql / postgresql+psycopg → postgresql+asyncpg
    if "+asyncpg" in url:
        return url
    if url.startswith("postgresql+psycopg://"):
        return url.replace("postgresql+psycopg://", "postgresql+asyncpg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url  # last resort


# Prefer explicit CI/local vars, then TEST_DB_URL
sync_url = os.getenv("DATABASE_URL_SYNC") or os.getenv("TEST_DB_URL")
async_url = os.getenv("DATABASE_URL") or (_to_asyncpg(sync_url) if sync_url else None)

if not (sync_url and async_url):
    raise RuntimeError(
        "Database URLs not set. Provide DATABASE_URL and DATABASE_URL_SYNC " "or a TEST_DB_URL (sync, psycopg)."
    )

SessionLocal = make_async_sessionmaker(async_url)
sink = make_universal_sink(SessionLocal, sync_url, "package_logger")
