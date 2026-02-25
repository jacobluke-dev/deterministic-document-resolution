from __future__ import annotations

from typing import Any

from sqlalchemy import func, select

from public_api.db.models import GlossaryEntry


class GlossaryRepository:
    """Read-only glossary access.

    Uses a SQLAlchemy query via DBManager.session() to guarantee a
    case-insensitive match regardless of DBManager's criteria syntax.
    """

    def __init__(self, *, dbm: Any) -> None:
        self._dbm = dbm

    def get(self, *, acronym: str) -> dict[str, Any] | None:
        if self._dbm is None:
            return None

        try:
            with self._dbm.session() as s:
                row = (
                    s.execute(
                        select(GlossaryEntry)
                        .where(func.lower(GlossaryEntry.acronym) == acronym.lower())
                        .limit(1)
                    )
                    .scalars()
                    .first()
                )
                if row is None:
                    return None
                return {
                    "acronym": row.acronym,
                    "definition": row.definition,
                    "source": row.source,
                }
        except Exception:
            # Fail closed: no enrichment rather than breaking /v1/resolve
            return None
