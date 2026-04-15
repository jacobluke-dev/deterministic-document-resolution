from __future__ import annotations

import os
import sys
from urllib.parse import urlsplit

from document_resolution_core.db_manager.dbm_factory import make_dbm
from public_api.core.settings import db_settings
from public_api.db.models import GlossaryAcronym, GlossaryVariant
from public_api.db.repos import SqlAlchemyAcronymRepo
from sqlalchemy import select


def main(force: bool = False) -> None:
    u = urlsplit(db_settings.database_url)
    print(f"DB host={u.hostname} port={u.port} db={u.path.lstrip('/')} user={u.username}")

    if os.getenv("APP_ENV", "local") != "local" and not force:
        sys.stderr.write("Refusing to seed: APP_ENV != local. Use --force to override.\n")
        sys.exit(2)

    if db_settings.DATABASE_DISABLED:
        sys.stderr.write("Refusing to seed: DATABASE_DISABLED=true.\n")
        sys.exit(2)

    # NOTE:
    # - Keep these mostly "paren-regex friendly" for current /v1/resolve demo.
    # - Avoid punctuation-heavy acronyms like "R&D" unless/until UN-70 pipeline integration is live.
    #
    # Structure:
    #   (ACRONYM, [(DOMAIN, DEFINITION), ...], PROVENANCE, [VARIANT, ...])
    examples: list[tuple[str, list[tuple[str, str]], str, list[str]]] = [
        ("MPS", [("general", "Metropolitan Police Service.")], "seed", ["Met Police", "Metropolitan Police"]),
        ("NHS", [("general", "UK National Health Service.")], "seed", ["National Health Service"]),
        ("GPU", [("general", "Graphics Processing Unit.")], "seed", ["Graphics Card"]),
        ("CPU", [("general", "Central Processing Unit.")], "seed", ["Processor"]),
        (
            "PDF",
            [
                ("general", "Portable Document Format."),
                # Uncomment when we want to demo ambiguity:
                # ("statistics", "Probability Density Function."),
            ],
            "seed",
            ["Portable Document Format"],
        ),
        ("GDPR", [("general", "General Data Protection Regulation.")], "seed", ["General Data Protection Regulation"]),
        ("ECHR", [("general", "European Convention on Human Rights.")], "seed",
         ["European Convention on Human Rights"]),
        ("FOI", [("general", "Freedom of Information.")], "seed", ["Freedom of Information"]),
        ("SAR", [("general", "Subject Access Request.")], "seed", ["Subject Access Request"]),
        ("SLA", [("general", "Service Level Agreement.")], "seed",
         ["Service-Level Agreement", "Service Level Agreements"]),
        ("KPI", [("general", "Key Performance Indicator.")], "seed", ["Key Performance Indicators"]),
    ]

    dbm = make_dbm(test_mode=False)

    repo = SqlAlchemyAcronymRepo(dbm=dbm)

    for acro, meanings, src, _ in examples:
        for domain, definition in meanings:
            repo.upsert_entry(
                acro,
                definition,
                domain=domain,
                source_ref=f"seed:{acro.lower()}:{domain}",
                provenance=src,
                is_active=True,
            )

    with dbm.session() as s:
        for acro, _, _, variants in examples:
            existing = (
                s.execute(
                    select(GlossaryAcronym).where(
                        GlossaryAcronym.tenant_id.is_(None),
                        GlossaryAcronym.normalized == acro.lower(),
                    )
                )
                .scalar_one()
            )

            # Idempotent variant insert (case-insensitive to match your unique index)
            existing_variants = {
                v.variant.lower() for v in getattr(existing, "variants", []) if getattr(v, "variant", None)
            }
            for v in variants:
                norm_v = v.lower()
                if norm_v not in existing_variants:
                    s.add(GlossaryVariant(acronym_id=existing.id,  variant=v))
                    existing_variants.add(norm_v)


if __name__ == "__main__":
    main(force="--force" in sys.argv)
