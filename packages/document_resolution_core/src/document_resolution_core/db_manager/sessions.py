from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


def to_asyncpg(url: str) -> str:
    """Convert a PostgreSQL SQLAlchemy URL to use the asyncpg driver.


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
    """Create an ``AsyncSession`` factory for the document_resolution database.


    Args:
      url: Database connection URL for the target PostgreSQL database.

    Returns:
      An async session factory producing ``AsyncSession`` instances bound to the
      configured engine.
    """
    engine = create_async_engine(
        url,
        pool_pre_ping=True,
        connect_args={"server_settings": {"search_path": "document_resolution"}},  # asyncpg only
    )
    return async_sessionmaker(engine, expire_on_commit=False)
