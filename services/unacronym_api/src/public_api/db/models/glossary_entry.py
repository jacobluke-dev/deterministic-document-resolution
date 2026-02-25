from __future__ import annotations

from typing import TYPE_CHECKING

from observability.db.models.base import BaseWithTimestamps
from sqlalchemy import Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

if TYPE_CHECKING:
    from .acronym_alias import AcronymAlias


class GlossaryEntry(BaseWithTimestamps):
    __tablename__ = "glossary_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    acronym: Mapped[str] = mapped_column(String(64), nullable=False)
    definition: Mapped[str] = mapped_column(Text, nullable=False)

    # NOTE: This is provenance/metadata (e.g. "seed", "import", "manual").
    # It is NOT a security boundary (public vs tenant) and must not be used as such.
    provenance: Mapped[str | None] = mapped_column(String(128), nullable=True)

    aliases: Mapped[list["AcronymAlias"]] = relationship(
        "AcronymAlias",
        back_populates="entry",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        # fast case-insensitive lookups and  prevent duplicates like "PDF"/"pdf"
        Index("ux_glossary_entries_lower_acronym", func.lower(acronym), unique=True),
    )
