from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from sqlalchemy import Integer, String, Text, TIMESTAMP, Identity
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base


class Logger(Base):
    __tablename__ = "logger"
    __table_args__ = {"schema": "logging"}

    id: Mapped[int] = mapped_column(Integer, Identity(always=False), primary_key=True)
    date_time: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)

    # core fields
    level_code: Mapped[int] = mapped_column(Integer, nullable=False)
    level_name: Mapped[str] = mapped_column(String(16), nullable=False)  # "debug"|"info"|...
    event: Mapped[str] = mapped_column(String(128), nullable=False)      # event name, e.g. "http_access"
    logger_type: Mapped[str] = mapped_column(String(32), nullable=False) # "decorator"|"api"|...

    # optional metadata
    function_name: Mapped[Optional[str]] = mapped_column(String(128))
    request_id: Mapped[Optional[str]] = mapped_column(String(64))
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer)

    # http-ish extras (optional)
    path: Mapped[Optional[str]] = mapped_column(String(512))
    method: Mapped[Optional[str]] = mapped_column(String(16))
    status: Mapped[Optional[int]] = mapped_column(Integer)
    bytes: Mapped[Optional[int]] = mapped_column(Integer)
    client_ip: Mapped[Optional[str]] = mapped_column(String(64))
    key_id: Mapped[Optional[str]] = mapped_column(String(128))

    # payloads
    info: Mapped[Optional[str]] = mapped_column(Text)                    # free-form summary (same as event if you like)
    arguments: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    keyword_arguments: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
