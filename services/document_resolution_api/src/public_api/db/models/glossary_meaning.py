from __future__ import annotations

from typing import TYPE_CHECKING

from observability.db.models.base import BaseWithTimestamps
from sqlalchemy import BigInteger, Boolean, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

if TYPE_CHECKING:
    from .glossary_acronym import GlossaryAcronym


class GlossaryMeaning(BaseWithTimestamps):
    __tablename__ = "glossary_meanings"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    acronym_id: Mapped[int] = mapped_column(
        ForeignKey("glossary_acronyms.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    definition: Mapped[str] = mapped_column(Text, nullable=False)
    domain: Mapped[str] = mapped_column(String(64), nullable=False, default="general")

    provenance: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source_ref: Mapped[str | None] = mapped_column(Text, nullable=True)

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    acronym: Mapped[GlossaryAcronym] = relationship(
        "GlossaryAcronym",
        back_populates="meaning",
    )

    __table_args__ = (
        Index("ux_glossary_meaning_acronym_domain", "acronym_id", "domain", unique=True),
        Index("ix_glossary_meaning_acronym_active", "acronym_id", "is_active"),
    )
