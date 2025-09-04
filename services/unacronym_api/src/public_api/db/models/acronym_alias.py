from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from public_api.db.models.base import BaseWithTimestamps

if TYPE_CHECKING:
    from .glossary_entry import GlossaryEntry

class AcronymAlias(BaseWithTimestamps):
    __tablename__ = "acronym_aliases"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    entry_id: Mapped[int] = mapped_column(
        ForeignKey("glossary_entries.id", ondelete="CASCADE"), nullable=False
    )
    alias: Mapped[str] = mapped_column(String(128), nullable=False)

    entry: Mapped[GlossaryEntry] = relationship(
        "GlossaryEntry",
        back_populates="aliases",
    )

    __table_args__ = (
        Index("ux_alias_entry_unique", entry_id, alias, unique=True),
    )
