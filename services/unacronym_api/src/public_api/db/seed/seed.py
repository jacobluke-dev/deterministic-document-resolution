from __future__ import annotations

import os
import sys
from urllib.parse import urlsplit

from plainera_core.db_manager.factory import make_dbm
from public_api.core.settings import db_settings
from public_api.db.models import AcronymAlias, GlossaryEntry
from sqlalchemy import func, select


def main(force: bool = False) -> None:
    u = urlsplit(db_settings.database_url)
    print(f"DB host={u.hostname} port={u.port} db={u.path.lstrip('/')} user={u.username}")

    if os.getenv("APP_ENV", "local") != "local" and not force:
        sys.stderr.write("Refusing to seed: APP_ENV != local. Use --force to override.\n")
        sys.exit(2)

    # NOTE:
    # - Keep these mostly "paren-regex friendly" for current /v1/resolve demo.
    # - Avoid punctuation-heavy acronyms like "R&D" unless/until UN-70 pipeline integration is live.
    examples: list[tuple[str, str, str, list[str]]] = [
        ("MPS", "Metropolitan Police Service.", "seed", ["Met Police", "Metropolitan Police"]),
        ("NHS", "UK National Health Service.", "seed", ["National Health Service"]),
        ("GPU", "Graphics Processing Unit.", "seed", ["Graphics Card"]),
        ("CPU", "Central Processing Unit.", "seed", ["Processor"]),
        ("PDF", "Portable Document Format.", "seed", ["Portable Document Format"]),
        ("GDPR", "General Data Protection Regulation.", "seed", ["General Data Protection Regulation"]),
        ("ECHR", "European Convention on Human Rights.", "seed", ["European Convention on Human Rights"]),
        ("FOI", "Freedom of Information.", "seed", ["Freedom of Information"]),
        ("SAR", "Subject Access Request.", "seed", ["Subject Access Request"]),
        ("SLA", "Service Level Agreement.", "seed", ["Service-Level Agreement", "Service Level Agreements"]),
        ("KPI", "Key Performance Indicator.", "seed", ["Key Performance Indicators"]),
        # Keep for later (won't match current demo regex):
        # ("R&D", "Research and Development.", "seed", ["Research & Development", "RnD"]),
    ]

    dbm = make_dbm(test_mode=False)

    with dbm.session() as s:
        for acro, defn, src, aliases in examples:
            # upsert-ish: match case-insensitively
            existing = (
                s.execute(
                    select(GlossaryEntry).where(func.lower(GlossaryEntry.acronym) == acro.lower())
                )
                .scalar_one_or_none()
            )

            if existing is None:
                existing = GlossaryEntry(acronym=acro, definition=defn, source=src)
                s.add(existing)
                s.flush()
            else:
                existing.definition = defn
                existing.source = src

            # Idempotent alias insert (case-sensitive; adjust if you want CI uniqueness)
            existing_aliases = {a.alias for a in getattr(existing, "aliases", [])}
            for al in aliases:
                if al not in existing_aliases:
                    s.add(AcronymAlias(entry_id=existing.id, alias=al))

        s.commit()


if __name__ == "__main__":
    main(force="--force" in sys.argv)
