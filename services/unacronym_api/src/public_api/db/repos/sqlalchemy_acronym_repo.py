from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from plainera_core.db_manager.connection import DBManager
from sqlalchemy import Select, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from public_api.db.models import GlossaryAcronym, GlossaryMeaning, GlossaryVariant
from public_api.db.repos.acronym_repo import AcronymRepo, GlossaryItem


class SqlAlchemyAcronymRepo(AcronymRepo):
    """SQLAlchemy-backed glossary repository."""

    def __init__(self, *, dbm: DBManager, default_limit: int = 10) -> None:
        """Initialise the repository.

        Args:
            dbm: Database manager used to create request-scoped sessions.
            default_limit: Fallback limit used when callers pass a non-positive
                limit.
        """
        self._dbm = dbm
        self._default_limit = default_limit

    @staticmethod
    def _normalize_domain(domain: str | None) -> str:
        """Normalise a domain value to the stored representation.

        Blank and ``None`` values are mapped to ``"general"``.

        Args:
            domain: Raw domain value supplied by a caller.

        Returns:
            Normalised domain string.
        """
        normalized = (domain or "").strip()
        return normalized or "general"

    def get_definitions(
        self,
        acronym: str,
        *,
        domain: str | None = None,
        limit: int = 10,
    ) -> list[GlossaryItem]:
        """Return active glossary entries for an acronym/domain pair.

        Matching is case-insensitive via the stored normalised acronym value.
        Results are restricted to global glossary rows and ordered
        deterministically by newest first.

        Args:
            acronym: Acronym to resolve.
            domain: Optional domain filter. ``None`` falls back to
                ``"general"``.
            limit: Maximum number of rows to return.

        Returns:
            Matching active glossary items.
        """
        normalized = self._normalize_required(acronym, field_name="acronym")
        normalized_domain = self._normalize_domain(domain)
        effective_limit = self._normalize_limit(limit, default=self._default_limit)

        with self._dbm.session() as session:
            stmt = (
                select(GlossaryMeaning, GlossaryAcronym)
                .join(GlossaryAcronym, GlossaryMeaning.acronym_id == GlossaryAcronym.id)
                .where(
                    GlossaryAcronym.tenant_id.is_(None),
                    GlossaryAcronym.is_active.is_(True),
                    GlossaryMeaning.is_active.is_(True),
                    GlossaryAcronym.normalized == normalized,
                    GlossaryMeaning.domain == normalized_domain,
                )
                .order_by(GlossaryMeaning.created_at.desc(), GlossaryMeaning.id.desc())
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
        """Insert or update a glossary meaning for an acronym/domain pair.

        The acronym identity is resolved case-insensitively against the global
        glossary. Meanings are unique per acronym/domain; an existing row is
        updated in place, otherwise a new row is inserted. On an integrity race,
        the operation is retried once.

        Args:
            acronym: Acronym surface form to persist.
            definition: Meaning text to store.
            domain: Optional domain value. ``None`` falls back to
                ``"general"``.
            provenance: High-level source label for the meaning.
            source_ref: Optional source breadcrumb for traceability.
            is_active: Whether the meaning should be active after the upsert.

        Returns:
            The inserted or updated glossary item.
        """
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
        """Return active glossary entries matched via a variant/alias.

        Alias matching is case-insensitive and restricted to global glossary
        rows. Results are filtered by domain and ordered deterministically by
        newest first.

        Args:
            alias: Alias or variant text to resolve.
            domain: Optional domain filter. ``None`` falls back to
                ``"general"``.
            limit: Maximum number of rows to return.

        Returns:
            Matching active glossary items.
        """
        normalized = self._normalize_required(alias, field_name="alias")
        effective_limit = self._normalize_limit(limit, default=self._default_limit)

        with self._dbm.session() as session:
            stmt = (
                select(GlossaryMeaning, GlossaryAcronym)
                .join(GlossaryAcronym, GlossaryMeaning.acronym_id == GlossaryAcronym.id)
                .join(GlossaryVariant, GlossaryVariant.acronym_id == GlossaryAcronym.id)
                .where(
                    GlossaryAcronym.tenant_id.is_(None),
                    GlossaryAcronym.is_active.is_(True),
                    GlossaryMeaning.is_active.is_(True),
                    func.lower(GlossaryVariant.variant) == normalized,
                )
                .order_by(GlossaryMeaning.created_at.desc(), GlossaryMeaning.id.desc())
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
        """List distinct active acronyms whose normalised form matches a prefix.

        Results are restricted to global glossary rows and to acronyms with at
        least one active meaning in the requested domain.

        Args:
            prefix: Case-insensitive acronym prefix to match.
            domain: Optional domain filter. ``None`` falls back to
                ``"general"``.
            limit: Maximum number of acronyms to return.

        Returns:
            Sorted distinct acronym surface forms.
        """
        normalized_prefix = (prefix or "").strip().lower()
        effective_limit = self._normalize_limit(limit, default=50)

        with self._dbm.session() as session:
            stmt = (
                select(GlossaryAcronym.acronym)
                .join(GlossaryMeaning, GlossaryMeaning.acronym_id == GlossaryAcronym.id)
                .where(
                    GlossaryAcronym.tenant_id.is_(None),
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
        """Soft-delete a glossary meaning by marking it inactive.

        The operation is idempotent: missing or already inactive rows are
        ignored.

        Args:
            entry_id: Primary key of the glossary meaning to deactivate.

        Raises:
            ValueError: If ``entry_id`` is not positive.
        """
        if entry_id <= 0:
            raise ValueError("entry_id must be positive")

        with self._dbm.session() as session:
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
        """Execute a single insert-or-update attempt.

        Args:
            normalized_acronym: Lower-cased acronym identity key.
            acronym: Canonical surface form to persist on the acronym row.
            definition: Meaning text to store.
            domain: Normalised domain value.
            provenance: High-level source label.
            source_ref: Optional source breadcrumb.
            is_active: Whether the resulting meaning should be active.

        Returns:
            The inserted or updated glossary item.
        """
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
        """Fetch or create the global acronym identity row.

        Args:
            session: Active SQLAlchemy session.
            acronym: Canonical surface form to persist if a new row is created.
            normalized_acronym: Lower-cased acronym identity key.

        Returns:
            Existing or newly created glossary acronym row.
        """
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
        stmt: Select[Any],
        *,
        domain: str | None,
    ) -> Select[Any]:
        """Apply the repository's normalised domain filter to a statement.

        Args:
            stmt: Statement selecting glossary meanings.
            domain: Optional domain filter supplied by the caller.

        Returns:
            Statement filtered to the normalised domain value.
        """
        return stmt.where(GlossaryMeaning.domain == self._normalize_domain(domain))

    @staticmethod
    def _to_item(*, meaning: GlossaryMeaning, acronym: GlossaryAcronym) -> GlossaryItem:
        """Convert ORM rows into the public repository DTO.

        Args:
            meaning: Glossary meaning ORM row.
            acronym: Parent glossary acronym ORM row.

        Returns:
            Immutable glossary item DTO.
        """
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
        """Validate and normalise a required string input.

        Args:
            value: Raw input string.
            field_name: Field name used in validation errors.
            lower: Whether to lower-case the normalised result.

        Returns:
            Trimmed, optionally lower-cased string.

        Raises:
            ValueError: If the value is blank after trimming.
        """
        normalized = (value or "").strip()
        if not normalized:
            raise ValueError(f"{field_name} must not be blank")
        return normalized.lower() if lower else normalized

    @staticmethod
    def _normalize_optional(value: str | None) -> str | None:
        """Normalise an optional string input.

        Args:
            value: Raw optional string.

        Returns:
            Trimmed string, or ``None`` if the input is ``None`` or blank.
        """
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @staticmethod
    def _normalize_limit(limit: int, *, default: int) -> int:
        """Return a safe positive query limit.

        Args:
            limit: Caller-supplied limit.
            default: Fallback value for non-positive limits.

        Returns:
            ``limit`` when positive, otherwise ``default``.
        """
        if limit <= 0:
            return default
        return limit

    @staticmethod
    def _utcnow() -> datetime:
        """Return the current UTC timestamp.

        Returns:
            Timezone-aware UTC datetime.
        """
        return datetime.now(timezone.utc)
