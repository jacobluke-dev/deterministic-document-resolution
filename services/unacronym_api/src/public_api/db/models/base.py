from datetime import datetime, timezone
from typing import ClassVar

from sqlalchemy import DateTime, Integer, MetaData
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from public_api.core.settings import db_settings


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=db_settings.NAMING_CONVENTION, schema=db_settings.DB_SCHEMA)

class BaseWithTimestamps(Base):
    __abstract__: ClassVar[bool] = True

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
