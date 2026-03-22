from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Select, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from plainera_core.db_manager.connection import DBManager
from public_api.db.models import GlossaryAcronym, GlossaryMeaning, GlossaryVariant
from public_api.db.repos.acronym_repo import GlossaryItem, AcronymRepo


class SqlAlchemyAcronymRepo(AcronymRepo):
    """SQLAlchemy-backed glossary repository."""

    def __init__(self, *, dbm: DBManager, default_limit: int = 10) -> None:
        self._dbm = dbm
        self._default_limit = default_limit

    @staticmethod
    def _normalize_domain(domain: str | None) -> str:
        normalized = (domain or "").strip()
        return normalized or "general"

    def get_definitions(
        self,
        acronym: str,
        *,
        domain: str | None = None,
        limit: int = 10,
    ) -> list[GlossaryItem]:
        normalized = self._normalize_required(acronym, field_name="acronym")
        normalized_domain = self._normalize_domain(domain)
        effective_limit = self._normalize_limit(limit, default=self._default_limit)

        with self._dbm.session() as session:
            stmt = (
                select(GlossaryMeaning, GlossaryAcronym)
                .join(GlossaryAcronym, GlossaryMeaning.acronym_id == GlossaryAcronym.id)
                .where(
                    GlossaryAcronym.is_active.is_(True),
                    GlossaryMeaning.is_active.is_(True),
                    GlossaryAcronym.normalized == normalized,
                GlossaryMeaning.domain == normalized_domain,
                )
            .order_by(GlossaryMeaning.created_at.desc())
                .limit(effective_limit)
            )

            rows = session.execute(stmt).all()
            return [self._to_item(meaning=row[0], acronym=row[1]) for row in rows]

    def upsert_entry(
        self,
        acronym: str,
        definition: str,
        *,
        domain: str | None,
        provenance: str,
        source_ref: str | None = None,
        is_active: bool = True,
    ) -> GlossaryItem:
        normalized_acronym = self._normalize_required(acronym, field_name="acronym")
        normalized_definition = self._normalize_required(definition, field_name="definition", lower=False)
        normalized_provenance = self._normalize_required(provenance, field_name="provenance", lower=False)
        normalized_source_ref = self._normalize_optional(source_ref)
        normalized_domain = self._normalize_domain(domain)

        try:
            return self._upsert_once(
                normalized_acronym=normalized_acronym,
                acronym=acronym.strip(),
                definition=normalized_definition,
                domain=normalized_domain,
                provenance=normalized_provenance,
                source_ref=normalized_source_ref,
                is_active=is_active,
            )
        except IntegrityError:
            return self._upsert_once(
                normalized_acronym=normalized_acronym,
                acronym=acronym.strip(),
                definition=normalized_definition,
                domain=normalized_domain,
                provenance=normalized_provenance,
                source_ref=normalized_source_ref,
                is_active=is_active,
            )

    def get_by_alias(
        self,
        alias: str,
        *,
        domain: str | None = None,
        limit: int = 10,
    ) -> list[GlossaryItem]:
        normalized = self._normalize_required(alias, field_name="alias")
        effective_limit = self._normalize_limit(limit, default=self._default_limit)

        with self._dbm.session() as session:
            stmt = (
                select(GlossaryMeaning, GlossaryAcronym)
                .join(GlossaryAcronym, GlossaryMeaning.acronym_id == GlossaryAcronym.id)
                .join(GlossaryVariant, GlossaryVariant.acronym_id == GlossaryAcronym.id)
                .where(
                    GlossaryAcronym.is_active.is_(True),
                    GlossaryMeaning.is_active.is_(True),
                    func.lower(GlossaryVariant.variant) == normalized,
                )
                .order_by(
                    GlossaryMeaning.created_at.desc(),
                )
                .limit(effective_limit)
            )
            stmt = self._apply_domain_filter(stmt, domain=domain)

            rows = session.execute(stmt).all()
            return [self._to_item(meaning=row[0], acronym=row[1]) for row in rows]

    def list_acronyms(
        self,
        prefix: str,
        *,
        domain: str | None = None,
        limit: int = 50,
    ) -> list[str]:
        normalized_prefix = (prefix or "").strip().lower()
        effective_limit = self._normalize_limit(limit, default=50)

        with self._dbm.session() as session:
            stmt = (
                select(GlossaryAcronym.acronym)
                .join(GlossaryMeaning, GlossaryMeaning.acronym_id == GlossaryAcronym.id)
                .where(
                    GlossaryAcronym.is_active.is_(True),
                    GlossaryMeaning.is_active.is_(True),
                    func.lower(GlossaryAcronym.acronym).like(f"{normalized_prefix}%"),
                )
                .distinct()
                .order_by(GlossaryAcronym.acronym.asc())
                .limit(effective_limit)
            )
            stmt = self._apply_domain_filter(stmt, domain=domain)

            return list(session.execute(stmt).scalars().all())

    def deactivate_entry(self, entry_id: int) -> None:
        if entry_id <= 0:
            raise ValueError("entry_id must be positive")

        with self._dbm.session() as session:
            with session.begin():
                meaning = session.get(GlossaryMeaning, entry_id)
                if meaning is None or meaning.is_active is False:
                    return

                meaning.is_active = False
                meaning.updated_at = self._utcnow()

    def _upsert_once(
        self,
        *,
        normalized_acronym: str,
        acronym: str,
        definition: str,
        domain: str,
        provenance: str,
        source_ref: str | None,
        is_active: bool,
    ) -> GlossaryItem:
        with self._dbm.session() as session:
            glossary_acronym = self._get_or_create_acronym(
                session=session,
                acronym=acronym,
                normalized_acronym=normalized_acronym,
            )

            meaning = (
                session.execute(
                    select(GlossaryMeaning)
                    .where(
                        GlossaryMeaning.acronym_id == glossary_acronym.id,
                        GlossaryMeaning.definition == definition,
                        GlossaryMeaning.domain == domain,
                    )
                    .limit(1)
                )
                .scalars()
                .first()
            )

            now = self._utcnow()

            if meaning is None:
                meaning = GlossaryMeaning(
                    acronym_id=glossary_acronym.id,
                    definition=definition,
                    domain=domain,
                    provenance=provenance,
                    source_ref=source_ref,
                    is_active=is_active,
                    created_at=now,
                    updated_at=now,
                )
                session.add(meaning)
                session.flush()
            else:
                meaning.provenance = provenance
                meaning.source_ref = source_ref
                meaning.is_active = is_active
                meaning.updated_at = now
                session.flush()

            return self._to_item(meaning=meaning, acronym=glossary_acronym)

    @staticmethod
    def _get_or_create_acronym(
        *,
        session: Session,
        acronym: str,
        normalized_acronym: str,
    ) -> GlossaryAcronym:
        glossary_acronym = (
            session.execute(
                select(GlossaryAcronym)
                .where(
                    GlossaryAcronym.tenant_id.is_(None),
                    GlossaryAcronym.normalized == normalized_acronym,
                )
                .limit(1)
            )
            .scalars()
            .first()
        )

        if glossary_acronym is not None:
            return glossary_acronym

        glossary_acronym = GlossaryAcronym(
            acronym=acronym,
            normalized=normalized_acronym,
            tenant_id=None,
            is_active=True,
        )
        session.add(glossary_acronym)
        session.flush()
        return glossary_acronym

    def _apply_domain_filter(
        self,
        stmt: Select,
        *,
        domain: str | None,
    ) -> Select:
        return stmt.where(GlossaryMeaning.domain == self._normalize_domain(domain))

    @staticmethod
    def _to_item(*, meaning: GlossaryMeaning, acronym: GlossaryAcronym) -> GlossaryItem:
        return GlossaryItem(
            id=int(meaning.id),
            acronym=acronym.acronym,
            definition=meaning.definition,
            domain=meaning.domain,
            source_ref=meaning.source_ref,
            provenance=meaning.provenance,
            is_active=bool(meaning.is_active),
            created_at=meaning.created_at,
            updated_at=meaning.updated_at,
        )

    @staticmethod
    def _normalize_required(value: str, *, field_name: str, lower: bool = True) -> str:
        normalized = (value or "").strip()
        if not normalized:
            raise ValueError(f"{field_name} must not be blank")
        return normalized.lower() if lower else normalized

    @staticmethod
    def _normalize_optional(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @staticmethod
    def _normalize_limit(limit: int, *, default: int) -> int:
        if limit <= 0:
            return default
        return limit


    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(timezone.utc)
