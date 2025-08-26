# models/mixins.py
from datetime import UTC, datetime

from sqlalchemy import Column, DateTime


class TimestampMixin:
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))
