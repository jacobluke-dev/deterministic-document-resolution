from observability.db.mixins import LoggerCommonMixin
from observability.db.models.base import Base
from sqlalchemy import Identity, Index, Integer
from sqlalchemy.orm import Mapped, mapped_column


class PackageLogger(Base, LoggerCommonMixin):
    __tablename__ = "unacronym_pkg_logger"
    id: Mapped[int] = mapped_column(Integer, Identity(always=False), primary_key=True)
    __table_args__ = (Index("ix_pkg_logger_event_datetime", "event", "date_time"),)
