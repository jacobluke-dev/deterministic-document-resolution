# unacronym_api/migrations/env.py
import os
from logging.config import fileConfig
from alembic import context
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool
from public_api.db.models import Base

config = context.config

ENV = (os.getenv("ENVIRONMENT") or "").upper()
if ENV in {"LOCAL", "LOCAL_PROD"}:
    load_dotenv()

# naming convention (as you already had)
Base.metadata.naming_convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}
target_metadata = Base.metadata

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

SCHEMA = "unacronym"

def include_object(obj, name, type_, reflected, compare_to):
    # Never autogenerate ops for the alembic version table (any schema)
    if type_ == "table" and name == "alembic_version":
        return False
    return True

def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url") or os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("No DB URL for offline migrations (set sqlalchemy.url or DATABASE_URL).")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        compare_server_default=True,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()

def _ensure_schema_and_version_table(conn) -> None:
    conn.exec_driver_sql(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}"')
    conn.exec_driver_sql(
        f'CREATE TABLE IF NOT EXISTS {SCHEMA}.alembic_version ('
        'version_num VARCHAR(32) PRIMARY KEY)'
    )

def run_migrations_online() -> None:
    connection = config.attributes.get("connection")

    if connection is not None:
        _ensure_schema_and_version_table(connection)
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            version_table_schema=SCHEMA,
            version_table="alembic_version",
            include_object=include_object,
        )
        with context.begin_transaction():
            context.run_migrations()
        return

    # Build engine from ini / env
    url = os.getenv("DATABASE_URL")
    if url:
        config.set_main_option("sqlalchemy.url", url)

    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        future=True,
    )

    with connectable.connect() as connection:
        _ensure_schema_and_version_table(connection)
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            version_table_schema=SCHEMA,
            version_table="alembic_version",
            include_object=include_object,
        )
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
