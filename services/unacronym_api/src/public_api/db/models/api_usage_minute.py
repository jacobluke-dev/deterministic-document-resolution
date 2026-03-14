from __future__ import annotations

from datetime import datetime

from observability.db.models.base import Base
from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from public_api.db.models import ApiKey

SCHEMA = "unacronym"


class ApiUsageMinute(Base):
    __tablename__ = "api_usage_minute"
    __table_args__ = (
        Index(
            "uq_api_usage_minute_api_key_id_minute_bucket",
            "api_key_id",
            "minute_bucket",
            unique=True,
        ),
        {"schema": SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    api_key_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(f"{SCHEMA}.api_keys.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    minute_bucket: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    request_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    api_key: Mapped["ApiKey"] = relationship(
        "ApiKey",
        back_populates="usage_minute",
    )
