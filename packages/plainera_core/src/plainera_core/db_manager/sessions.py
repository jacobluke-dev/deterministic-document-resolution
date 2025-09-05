from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


def make_async_sessionmaker(url: str):
    engine = create_async_engine(
        url,
        pool_pre_ping=True,
        connect_args={"server_settings": {"search_path": "unacronym"}},  # asyncpg only
    )
    return async_sessionmaker(engine, expire_on_commit=False)
