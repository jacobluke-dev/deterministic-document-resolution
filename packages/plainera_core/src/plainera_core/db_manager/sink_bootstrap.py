import os

from plainera_core.db_manager.sessions import _to_asyncpg, make_async_sessionmaker
from plainera_core.db_manager.sink_factory import make_universal_sink
from plainera_core.db_manager.sinks import UniversalSink


def build_sink_from_env(name: str) -> UniversalSink:
    sync_url = os.getenv("DATABASE_URL_SYNC") or os.getenv("TEST_DB_URL")
    async_url = os.getenv("DATABASE_URL") or (_to_asyncpg(sync_url) if sync_url else None)

    if not (sync_url and async_url):
        raise RuntimeError(
            "Database URLs not set. Provide DATABASE_URL and DATABASE_URL_SYNC "
            "or a TEST_DB_URL."
        )

    session_local = make_async_sessionmaker(async_url)
    return make_universal_sink(session_local, sync_url, name)
