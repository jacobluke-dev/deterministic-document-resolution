from __future__ import annotations

from datetime import datetime

from observability.db.models.base import BaseWithTimestamps
from sqlalchemy import BigInteger, Boolean, DateTime, String, Text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

SCHEMA = "unacronym"


class ApiKey(BaseWithTimestamps):
    __tablename__ = "api_keys"
    __table_args__ = {"schema": SCHEMA}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    user_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    key_id: Mapped[str] = mapped_column(String(24), nullable=False, unique=True)
    key_hash: Mapped[str] = mapped_column(String(255), nullable=False)

    name: Mapped[str | None] = mapped_column(Text, nullable=True)
    prefix: Mapped[str] = mapped_column(String(10), nullable=False)

    scopes: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    last_used_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
