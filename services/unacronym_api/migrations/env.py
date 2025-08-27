import os
from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

from public_api.db.models import Base, BaseWithTimestamps, AcronymAlias, GlossaryEntry

config = context.config

# Load .env only in local modes (avoid TEST/CI at import time)
ENV = (os.getenv("ENVIRONMENT") or "").upper()
if ENV in {"LOCAL", "LOCAL_PROD"}:
    load_dotenv()

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}
Base.metadata.naming_convention = NAMING_CONVENTION
target_metadata = Base.metadata

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    if not url:
        env_url = os.getenv("DATABASE_URL", "")
        if env_url:
            config.set_main_option("sqlalchemy.url", env_url)
            url = env_url
    if not url:
        raise RuntimeError("No DB URL for offline migrations (set sqlalchemy.url or DATABASE_URL).")

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    # ✅ Respect a connection injected by tests:
    connection = config.attributes.get("connection")

    if connection is not None:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()
        return

    # Fallback: build an engine from .ini (or env var override)
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        config.set_main_option("sqlalchemy.url", env_url)

    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        future=True,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
