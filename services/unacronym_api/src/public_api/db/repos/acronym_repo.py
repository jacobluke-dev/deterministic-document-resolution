from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class GlossaryItem:
    id: int
    acronym: str
    definition: str
    domain: str
    provenance: str | None
    source_ref: str | None
    is_active: bool
    created_at: datetime
    updated_at: datetime


class AcronymRepo(ABC):
    @abstractmethod
    def get_definitions(
        self,
        acronym: str,
        *,
        domain: str | None = None,
        limit: int = 10,
    ) -> list[GlossaryItem]:
        raise NotImplementedError

    @abstractmethod
    def upsert_entry(
        self,
        acronym: str,
        definition: str,
        *,
        domain: str | None,
        source: str,
        is_active: bool = True,
    ) -> GlossaryItem:
        raise NotImplementedError

    @abstractmethod
    def deactivate_entry(self, entry_id: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_acronyms(
        self,
        prefix: str,
        *,
        domain: str | None = None,
        limit: int = 50,
    ) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def get_by_alias(
        self,
        alias: str,
        *,
        domain: str | None = None,
        limit: int = 10,
    ) -> list[GlossaryItem]:
        raise NotImplementedError
