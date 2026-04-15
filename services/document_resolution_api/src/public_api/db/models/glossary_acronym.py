from __future__ import annotations

from typing import TYPE_CHECKING

from observability.db.models.base import BaseWithTimestamps
from sqlalchemy import BigInteger, Boolean, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

if TYPE_CHECKING:
    from .glossary_meaning import GlossaryMeaning
    from .glossary_variant import GlossaryVariant


class GlossaryAcronym(BaseWithTimestamps):
    __tablename__ = "glossary_acronyms"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    # Reserved for later multi-tenant. Keep nullable for now.
    tenant_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    acronym: Mapped[str] = mapped_column(String(64), nullable=False)
    normalized: Mapped[str] = mapped_column(String(64), nullable=False)

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    meaning: Mapped[list[GlossaryMeaning]] = relationship(
        "GlossaryMeaning",
        back_populates="acronym",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    variants: Mapped[list[GlossaryVariant]] = relationship(
        "GlossaryVariant",
        back_populates="acronym",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        Index(
            "ux_glossary_acronyms_global_normalized",
            "normalized",
            unique=True,
            postgresql_where=(tenant_id.is_(None)),
        ),
        Index(
            "ux_glossary_acronyms_tenant_normalized",
            "tenant_id",
            "normalized",
            unique=True,
            postgresql_where=(tenant_id.is_not(None)),
        ),
        Index("ix_glossary_acronyms_normalized", "normalized"),
        Index(
            "ix_glossary_acronyms_active_normalized",
            "normalized",
            postgresql_where=(is_active.is_(True)),
        ),
    )
