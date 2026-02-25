from __future__ import annotations

from typing import Any


class GlossaryRepository:
    """
    Thin repository wrapper around the existing DBManager.

    Notes:
      - Keeps table/criteria details out of routers/services.
      - Uses the same selection semantics as your current implementation to avoid surprises.
      - Returns raw dict row data (service maps into GlossaryMatch).
    """

    def __init__(self, *, dbm: Any, table_fqn: str = "glossary_entries") -> None:
        self._dbm = dbm
        self._table = table_fqn

    def get(self, *, acronym: str) -> dict[str, Any] | None:
        if self._dbm is None:
            return None

        try:
            ac = acronym.lower()

            row = self._dbm.select_one_dict(
                table_fqn=self._table,
                columns=["acronym", "definition", "source"],
                criteria=[("acronym", "", ac)],
            )
            return row or None
        except Exception:
            # Fail closed (no glossary enrichment) rather than taking down /v1/resolve.
            return None
