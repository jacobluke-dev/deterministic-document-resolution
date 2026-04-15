from __future__ import annotations

from datetime import date

from observability.db.models.base import BaseWithTimestamps
from sqlalchemy import BigInteger, Date, ForeignKey, Index, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from public_api.db.models import ApiKey

SCHEMA = "document_resolution"


class ApiUsageDaily(BaseWithTimestamps):
    __tablename__ = "api_usage_daily"
    __table_args__ = (
        Index(
            "uq_api_usage_daily_api_key_id_usage_date",
            "api_key_id",
            "usage_date",
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

    usage_date: Mapped[date] = mapped_column(Date, nullable=False)
    request_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    api_key: Mapped[ApiKey] = relationship(
        "ApiKey",
        back_populates="usage_daily",
    )
