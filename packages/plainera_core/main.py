# app_bootstrap.py (or wherever you init dependencies)
from db_manager import PostgresDbSink, make_async_sessionmaker
from observability.observability.decorator import logger

SessionLocal = make_async_sessionmaker("postgresql+asyncpg://user:pass@localhost:5432/test_db")
db_sink = PostgresDbSink(SessionLocal)


@logger(message="demo:add", arg_names=["a","b"], log_result=True, db_sink=db_sink)
async def add(a: int, b: int) -> int:
    return a + b

if __name__ == "__main__":
    add(1, 2)
