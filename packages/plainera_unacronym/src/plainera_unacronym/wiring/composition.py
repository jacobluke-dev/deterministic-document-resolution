import os

from dotenv import load_dotenv

load_dotenv()
from plainera_core.db_manager.sessions import make_async_sessionmaker
from plainera_core.db_manager.sink_factory import make_universal_sink

ASYNC_URL = os.environ["DATABASE_URL"]  # postgresql+asyncpg://...
SYNC_URL  = os.environ["DATABASE_URL_SYNC"]   # postgresql+psycopg://...

SessionLocal = make_async_sessionmaker(ASYNC_URL)
sink = make_universal_sink(SessionLocal, SYNC_URL, "package_logger")
