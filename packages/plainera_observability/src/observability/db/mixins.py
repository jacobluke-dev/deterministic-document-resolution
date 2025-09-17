from datetime import datetime
from typing import Any
from sqlalchemy import Integer, String, Text, TIMESTAMP, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

class LoggerCommonMixin:
    """Abstract mixin for logger columns. No table, no Base."""
    __abstract__ = True

    # core fields
    level_code: Mapped[int] = mapped_column(Integer, nullable=False)
    level_name: Mapped[str] = mapped_column(String(16), nullable=False)
    event:      Mapped[str] = mapped_column(String(128), nullable=False)
    logger_type:Mapped[str] = mapped_column(String(32),  nullable=False)

    # optional metadata
    function_name: Mapped[str | None] = mapped_column(String(128))
    request_id:    Mapped[str | None] = mapped_column(String(64))
    duration_ms:   Mapped[int | None] = mapped_column(Integer)

    # payloads
    info:               Mapped[str | None] = mapped_column(Text)
    arguments:          Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    keyword_arguments:  Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    # timestamps
    date_time: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
