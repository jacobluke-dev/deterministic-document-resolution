from __future__ import annotations

from typing import Any

from sqlalchemy import func, select

from public_api.db.models import GlossaryAcronym, GlossaryMeaning, GlossaryVariant


class GlossaryRepository:
    """Read-only glossary access (normalised schema)."""

    def __init__(self, *, dbm: Any) -> None:
        self._dbm = dbm

    def get(self, *, acronym: str, domain: str | None = None) -> dict[str, Any] | None:
        if self._dbm is None:
            return None

        norm = acronym.lower()
        dom = domain or "general"

        try:
            with self._dbm.session() as s:
                # 1) resolve acronym identity (direct)
                ga = (
                    s.execute(
                        select(GlossaryAcronym)
                        .where(
                            GlossaryAcronym.tenant_id.is_(None),
                            GlossaryAcronym.normalized == norm,
                            GlossaryAcronym.is_active.is_(True),
                        )
                        .limit(1)
                    )
                    .scalars()
                    .first()
                )

                # 2) resolve via variant if needed
                if ga is None:
                    ga = (
                        s.execute(
                            select(GlossaryAcronym)
                            .join(GlossaryVariant, GlossaryVariant.acronym_id == GlossaryAcronym.id)
                            .where(
                                GlossaryAcronym.tenant_id.is_(None),
                                GlossaryAcronym.is_active.is_(True),
                                func.lower(GlossaryVariant.variant) == norm,
                            )
                            .limit(1)
                        )
                        .scalars()
                        .first()
                    )

                if ga is None:
                    return None

                # 3) choose meaning (domain → else general)
                meaning = (
                    s.execute(
                        select(GlossaryMeaning)
                        .where(
                            GlossaryMeaning.acronym_id == ga.id,
                            GlossaryMeaning.domain == dom,
                            GlossaryMeaning.is_active.is_(True),
                        )
                        .limit(1)
                    )
                    .scalars()
                    .first()
                )

                # Fallback to general if caller asked for a domain that doesn't exist
                if meaning is None and dom != "general":
                    meaning = (
                        s.execute(
                            select(GlossaryMeaning)
                            .where(
                                GlossaryMeaning.acronym_id == ga.id,
                                GlossaryMeaning.domain == "general",
                                GlossaryMeaning.is_active.is_(True),
                            )
                            .limit(1)
                        )
                        .scalars()
                        .first()
                    )

                if meaning is None:
                    return None

                return {
                    "acronym": ga.acronym,
                    "definition": meaning.definition,
                    "provenance": meaning.provenance,
                }
        except Exception:
            # Fail closed: no enrichment rather than breaking /v1/resolve
            return None
