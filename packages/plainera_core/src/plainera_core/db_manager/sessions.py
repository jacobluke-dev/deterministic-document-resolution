from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


def _to_asyncpg(url: str) -> str:
    # handle postgresql / postgresql+psycopg → postgresql+asyncpg
    if "+asyncpg" in url:
        return url
    if url.startswith("postgresql+psycopg://"):
        return url.replace("postgresql+psycopg://", "postgresql+asyncpg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url  # last resort



def make_async_sessionmaker(url: str) -> async_sessionmaker[AsyncSession]:
    engine = create_async_engine(
        url,
        pool_pre_ping=True,
        connect_args={"server_settings": {"search_path": "unacronym"}},  # asyncpg only
    )
    return async_sessionmaker(engine, expire_on_commit=False)
