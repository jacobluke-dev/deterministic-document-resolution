from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import TIMESTAMP, Identity, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Logger(Base):
    __tablename__ = "logger"

    id: Mapped[int] = mapped_column(Integer, Identity(always=False), primary_key=True)

    # core fields
    level_code: Mapped[int] = mapped_column(Integer, nullable=False)
    level_name: Mapped[str] = mapped_column(String(16), nullable=False)  # "debug"|"info"|...
    event: Mapped[str] = mapped_column(String(128), nullable=False)      # event name, e.g. "http_access"
    logger_type: Mapped[str] = mapped_column(String(32), nullable=False) # "decorator"|"api"|...

    # optional metadata
    function_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # http-ish extras (optional)
    path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    method: Mapped[str | None] = mapped_column(String(16), nullable=True)
    status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    client_ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    key_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # payloads
    info: Mapped[str | None] = mapped_column(Text)                    # free-form summary (same as event if you like)
    arguments: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    keyword_arguments: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)

    date_time: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
