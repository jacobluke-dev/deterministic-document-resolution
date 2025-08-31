# app_bootstrap.py (or wherever you init dependencies)
from db.models import Logger
from db_manager.mappers import default_map
from db_manager.sessions import make_async_sessionmaker
from db_manager.sinks import SqlAlchemyModelSink
from observability.observability.decorator import logger

SessionLocal = make_async_sessionmaker("postgresql+asyncpg://user:pass@localhost:5432/test_db")
db_sink = SqlAlchemyModelSink(SessionLocal, Logger, default_map)


@logger(message="demo:add", arg_names=["a","b"], log_result=True, db_sink=db_sink)
async def add(a: int, b: int) -> int:
    return a + b

if __name__ == "__main__":
    add(1, 2)
