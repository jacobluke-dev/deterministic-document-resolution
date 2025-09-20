from datetime import datetime, timezone

from public_api.core.settings import db_settings
from sqlalchemy import DateTime, Integer, MetaData, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=db_settings.NAMING_CONVENTION, schema=db_settings.DB_SCHEMA)


class BaseWithTimestamps(Base):
    __abstract__ = True

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),  # DB default
        default=lambda: datetime.now(timezone.utc),  # client default (safe fallback)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),  # initial insert default at DB
        onupdate=lambda: datetime.now(timezone.utc),  # app-driven updates
        # If you add a DB trigger for updated_at, you can drop onupdate.
    )
