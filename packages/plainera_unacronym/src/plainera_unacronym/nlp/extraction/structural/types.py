from __future__ import annotations

from dataclasses import dataclass

from plainera_unacronym.nlp.common.types import Span
from plainera_unacronym.nlp.detection.structural.types import StructuralReferenceKind


@dataclass(frozen=True)
class StructuralReferenceEntry:
    kind: StructuralReferenceKind
    label: str
    canonical_label: str
    normalized_key: str
    canonical_key: str
    start_offset: int
    end_offset: int
    provenance: str


@dataclass(frozen=True)
class StructuralReferenceLink:
    kind: StructuralReferenceKind
    label: str
    canonical_label: str
    normalized_key: str
    canonical_key: str
    reference_span: Span
    target_span: Span | None
    confidence: float
    provenance: str

@dataclass(frozen=True)
class StructuralAnchor:
    label: str
    normalized_key: str
    start_offset: int
    end_offset: int
    ordinal: int


@dataclass(frozen=True)
class StructuralReferenceResolutionResult:
    references: list[StructuralReferenceEntry]
    links: list[StructuralReferenceLink]
    unique_keys: dict[str, StructuralReferenceEntry]
    unique_links: dict[str, StructuralReferenceLink]
