import os, uuid
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from plainera_core.db_manager.connection import DBManager


def _base_url() -> str:
    return os.environ["DATABASE_URL_BASE"]  # WITHOUT /dbname

def _default_db() -> str:
    return os.environ.get("POSTGRES_DB", "postgres")

def _make_engine(dbname: str):
    return create_engine(f"{_base_url()}/{dbname}", future=True)

def create_temp_database(prefix: str = "temp") -> tuple[str, DBManager]:
    name = f"{prefix}_{uuid.uuid4().hex[:12]}"
    admin = _make_engine(_default_db())
    with admin.begin() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    eng = _make_engine(name)
    sess = sessionmaker(bind=eng, future=True)
    return name, DBManager(eng, sess)

def drop_database(name: str) -> None:
    admin = _make_engine(_default_db())
    # terminate connections + drop
    with admin.begin() as conn:
        conn.execute(text("""
          SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = :db
        """), {"db": name})
        conn.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))

def create_temp_schema(dbm: DBManager, prefix: str = "tmp") -> str:
    schema = f'{prefix}_{uuid.uuid4().hex[:8]}'
    dbm.create_schema(schema)
    return schema
