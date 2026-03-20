from __future__ import annotations

import os
import sys
from urllib.parse import urlsplit

from plainera_core.db_manager.factory import make_dbm
from public_api.core.settings import db_settings
from public_api.db.models import GlossaryAcronym, GlossaryMeaning, GlossaryVariant
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
                # Uncomment when you want to demo ambiguity:
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

    with dbm.session() as s:
        for acro, meanings, src, variants in examples:
            norm = acro.lower()

            # Upsert-ish: acronym identity (case-insensitive), tenant_id is NULL for now.
            existing = (
                s.execute(
                    select(GlossaryAcronym).where(
                        GlossaryAcronym.tenant_id.is_(None),
                        GlossaryAcronym.normalized == norm,
                    )
                )
                .scalar_one_or_none()
            )

            if existing is None:
                existing = GlossaryAcronym(
                    tenant_id=None,
                    acronym=acro,
                    normalized=norm,
                    is_active=True,
                )
                s.add(existing)
                s.flush()
            else:
                # Keep canonical surface as last-seen seed value (optional).
                existing.acronym = acro
                existing.normalized = norm
                existing.is_active = True

            # Upsert meanings by (acronym_id, domain)
            for domain, definition in meanings:
                meaning = (
                    s.execute(
                        select(GlossaryMeaning).where(
                            GlossaryMeaning.acronym_id == existing.id,
                            GlossaryMeaning.domain == domain,
                        )
                    )
                    .scalar_one_or_none()
                )

                if meaning is None:
                    s.add(
                        GlossaryMeaning(
                            acronym_id=existing.id,
                            domain=domain,
                            definition=definition,
                            provenance=src,
                            is_active=True,
                        )
                    )
                else:
                    meaning.definition = definition
                    meaning.provenance = src
                    meaning.is_active = True

            # Idempotent variant insert (case-insensitive to match your unique index)
            existing_variants = {
                v.variant.lower() for v in getattr(existing, "variants", []) if getattr(v, "variant", None)
            }
            for v in variants:
                if v.lower() not in existing_variants:
                    s.add(GlossaryVariant(acronym_id=existing.id, variant=v))

        s.commit()


if __name__ == "__main__":
    main(force="--force" in sys.argv)
