from __future__ import annotations

from typing import TYPE_CHECKING

from observability.db.models.base import BaseWithTimestamps
from sqlalchemy import BigInteger, ForeignKey, Index, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

if TYPE_CHECKING:
    from .glossary_acronym import GlossaryAcronym


class GlossaryVariant(BaseWithTimestamps):
    __tablename__ = "glossary_variants"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    acronym_id: Mapped[int] = mapped_column(
        ForeignKey("glossary_acronyms.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    variant: Mapped[str] = mapped_column(String(128), nullable=False)

    acronym: Mapped[GlossaryAcronym] = relationship(
        "GlossaryAcronym",
        back_populates="variants",
    )

    __table_args__ = (
        Index(
            "ux_glossary_variant_acronym_lower_variant",
            "acronym_id",
            func.lower(variant),
            unique=True,
        ),
        Index("ix_glossary_variant_lower_variant", func.lower(variant)),
    )
