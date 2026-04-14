import os

from plainera_core.db_manager.sessions import make_async_session_maker, to_asyncpg
from plainera_core.db_manager.sink_factory import make_universal_sink
from plainera_core.db_manager.sinks import UniversalSink


def build_sink_from_env(name: str) -> UniversalSink:
    """Build a universal database sink from environment configuration.

    This reads synchronous and asynchronous database URLs from environment
    variables and constructs a ``UniversalSink`` for the named sink target.

    Environment resolution order:
      - Synchronous URL:
        - ``DATABASE_URL_SYNC``
        - ``TEST_DB_URL``
      - Asynchronous URL:
        - ``DATABASE_URL``
        - derived from the synchronous URL via ``to_asyncpg(...)``

    The function requires both a synchronous URL and an asynchronous URL. If
    only a synchronous URL is provided, the asynchronous variant is derived
    automatically. If the required configuration is incomplete, a
    ``RuntimeError`` is raised.

    Args:
      name: Logical sink name used when constructing the universal sink.

    Returns:
      A configured ``UniversalSink`` using an async session factory and sync
      database URL.

    Raises:
      RuntimeError: If neither the required sync nor async database URL
        configuration can be resolved from the environment.
    """
    sync_url = os.getenv("DATABASE_URL_SYNC") or os.getenv("TEST_DB_URL")
    async_url = os.getenv("DATABASE_URL") or (to_asyncpg(sync_url) if sync_url else None)

    if not (sync_url and async_url):
        raise RuntimeError(
            "Database URLs not set. Provide DATABASE_URL and DATABASE_URL_SYNC "
            "or a TEST_DB_URL."
        )

    session_local = make_async_session_maker(async_url)
    return make_universal_sink(session_local, sync_url, name)
