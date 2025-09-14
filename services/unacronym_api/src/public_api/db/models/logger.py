from __future__ import annotations
from sqlalchemy import Integer, Identity, String, CheckConstraint, Index
from sqlalchemy.orm import Mapped, mapped_column
from observability.db.mixins import LoggerCommonMixin
from observability.db.models.base import Base


class Logger(Base, LoggerCommonMixin):
    __tablename__ = "unacronym_api_logger"

    id: Mapped[int] = mapped_column(Integer, Identity(always=False), primary_key=True)

    path:     Mapped[str | None] = mapped_column(String(512))
    method:   Mapped[str | None] = mapped_column(String(16))
    status:   Mapped[int | None] = mapped_column(Integer)
    bytes:    Mapped[int | None] = mapped_column(Integer)
    client_ip:Mapped[str | None] = mapped_column(String(64))
    key_id:   Mapped[str | None] = mapped_column(String(128))

    __table_args__ = (
        CheckConstraint("status IS NULL OR (status BETWEEN 100 AND 599)", name="ck_logger_http_status"),
        Index("ix_logger_event_datetime", "event", "date_time"),
        Index("ix_logger_reqid_datetime", "request_id", "date_time"),
    )
