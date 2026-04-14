from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


def to_asyncpg(url: str) -> str:
    """Convert a PostgreSQL SQLAlchemy URL to use the asyncpg driver.

    This normalises supported PostgreSQL connection URL variants so they can be
    used with SQLAlchemy's async engine creation. If the URL already targets
    ``asyncpg``, it is returned unchanged.

    Supported rewrites:
      - ``postgresql+psycopg://`` -> ``postgresql+asyncpg://``
      - ``postgresql://`` -> ``postgresql+asyncpg://``

    Any non-PostgreSQL or unrecognised URL is returned unchanged as a fallback.

    Args:
      url: Database connection URL.

    Returns:
      A connection URL using the ``postgresql+asyncpg://`` dialect when the
      input matches a supported PostgreSQL variant; otherwise the original URL.
    """
    if "+asyncpg" in url:
        return url
    if url.startswith("postgresql+psycopg://"):
        return url.replace("postgresql+psycopg://", "postgresql+asyncpg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url  # last resort



def make_async_session_maker(url: str) -> async_sessionmaker[AsyncSession]:
    """Create an ``AsyncSession`` factory for the Unacronym database.

    This builds a SQLAlchemy async engine using ``asyncpg``-compatible settings
    and returns an ``async_sessionmaker`` configured with
    ``expire_on_commit=False``. The engine enables connection liveness checks
    via ``pool_pre_ping`` and sets the PostgreSQL ``search_path`` to the
    ``unacronym`` schema at connection time.

    Note:
      ``connect_args={"server_settings": {"search_path": "unacronym"}}`` is
      specific to the ``asyncpg`` driver. Callers should ensure the supplied
      URL is already using ``postgresql+asyncpg://`` or is normalised before
      engine creation.

    Args:
      url: Database connection URL for the target PostgreSQL database.

    Returns:
      An async session factory producing ``AsyncSession`` instances bound to the
      configured engine.
    """
    engine = create_async_engine(
        url,
        pool_pre_ping=True,
        connect_args={"server_settings": {"search_path": "unacronym"}},  # asyncpg only
    )
    return async_sessionmaker(engine, expire_on_commit=False)
