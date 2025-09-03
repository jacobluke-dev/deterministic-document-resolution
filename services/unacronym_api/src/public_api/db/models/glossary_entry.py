from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Index, String, Text, func, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.public_api.db.models.base import BaseWithTimestamps

if TYPE_CHECKING:
    from .acronym_alias import AcronymAlias

class GlossaryEntry(BaseWithTimestamps):
    __tablename__ = "glossary_entries"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    acronym: Mapped[str] = mapped_column(String(64), nullable=False)
    definition: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str | None] = mapped_column(String(128), nullable=True)

    aliases: Mapped[list["AcronymAlias"]] = relationship(
        "AcronymAlias",
        back_populates="entry",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_glossary_entries_lower_acronym", func.lower(acronym)),
    )
