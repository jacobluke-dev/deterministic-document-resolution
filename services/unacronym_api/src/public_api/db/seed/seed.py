from __future__ import annotations

import os
import sys
from urllib.parse import urlsplit

from db_manager.factory import make_dbm
from src.public_api.core.settings import db_settings
from src.public_api.db.models import AcronymAlias, GlossaryEntry
from sqlalchemy import select


def main(force: bool = False) -> None:
    u = urlsplit(db_settings.database_url)
    print(f"DB host={u.hostname} port={u.port} db={u.path.lstrip('/')} user={u.username}")
    if os.getenv("APP_ENV", "local") != "local" and not force:
        sys.stderr.write("Refusing to seed: APP_ENV != local. Use --force to override.\n")
        sys.exit(2)

    examples = [
        ("NHS", "UK's National Health Service.", "seed", ["National Health Service"]),
        ("GPU", "Graphics Processing Unit.", "seed", ["Graphics Card"]),
        ("R&D", "Research and Development.", "seed", ["Research & Development", "RnD"]),
    ]
    dbm = make_dbm()

    with dbm.session() as s:
        for acro, defn, src, aliases in examples:
            # upsert-ish: find by lower(acronym)
            existing = s.execute(select(GlossaryEntry).where(GlossaryEntry.acronym.ilike(acro))).scalar_one_or_none()
            if existing is None:
                existing = GlossaryEntry(acronym=acro, definition=defn, source=src)
                s.add(existing)
                s.flush()
            else:
                existing.definition = defn  # keep fresh
                existing.source = src

            for al in aliases:
                if not any(a.alias == al for a in existing.aliases):
                    s.add(AcronymAlias(entry_id=existing.id, alias=al))

if __name__ == "__main__":
    main(force="--force" in sys.argv)
