import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool
from src.public_api.db.models import Base
#registering tables...
import src.public_api.db.models.glossary_entry
import src.public_api.db.models.acronym_alias
import src.public_api.db.models.logger


config = context.config


def _normalize(url: str | None) -> str:
    if not url:
        return ""
    return url.replace("postgresql+psycopg2://", "postgresql+psycopg://")


ENV = (os.getenv("ENVIRONMENT") or "").upper()
if ENV in {"LOCAL", "LOCAL_PROD"}:
    from dotenv import load_dotenv
    load_dotenv()

Base.metadata.naming_convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

url = _normalize(os.getenv("DATABASE_URL") or config.get_main_option("sqlalchemy.url"))
if not url:
    raise RuntimeError("No database URL. Set DATABASE_URL or sqlalchemy.url")
config.set_main_option("sqlalchemy.url", url)

SCHEMA = "unacronym"
target_metadata = Base.metadata

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def _ensure_schema_and_version(conn) -> None:
    conn.exec_driver_sql(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}"')
    conn.exec_driver_sql(
        f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.alembic_version (
            version_num VARCHAR(32) PRIMARY KEY
        )
        """
    )


def _include_object(obj, name, type_, reflected, compare_to):
    if type_ == "table" and name == "alembic_version":
        return False
    return True


def run_migrations_offline() -> None:
    url_offline = _normalize(config.get_main_option("sqlalchemy.url") or os.getenv("DATABASE_URL"))
    if not url_offline:
        raise RuntimeError("No DB URL for offline migrations.")

    context.configure(
        url=url_offline,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        compare_server_default=True,
        include_schemas=True,
        version_table="alembic_version",
        version_table_schema=SCHEMA,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    injected = config.attributes.get("connection")
    if injected is not None:
        # IMPORTANT: do NOT use `with ... as ac:` which closes the underlying connection.
        ac = injected.execution_options(isolation_level="AUTOCOMMIT")
        _ensure_schema_and_version(ac)

        context.configure(
            connection=injected,
            target_metadata=target_metadata,
            include_schemas=True,
            compare_type=True,
            compare_server_default=True,
            version_table="alembic_version",
            version_table_schema=SCHEMA,
            include_object=_include_object,
            transactional_ddl=False,
        )
        with context.begin_transaction():
            context.run_migrations()
        return

    connectable = engine_from_config(
        {"sqlalchemy.url": url},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        future=True,
    )
    with connectable.connect() as connection:
        ac = connection.execution_options(isolation_level="AUTOCOMMIT")
        _ensure_schema_and_version(ac)

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_schemas=True,
            compare_type=True,
            compare_server_default=True,
            version_table="alembic_version",
            version_table_schema=SCHEMA,
            include_object=_include_object,
            transactional_ddl=False,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
