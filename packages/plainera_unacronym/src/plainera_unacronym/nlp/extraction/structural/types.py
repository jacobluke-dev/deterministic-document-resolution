from __future__ import annotations

from dataclasses import dataclass

from plainera_unacronym.nlp.detection.structural.types import StructuralReferenceKind


@dataclass(frozen=True)
class StructuralReferenceResolution:
    kind: StructuralReferenceKind
    label: str
    canonical_label: str
    normalized_key: str
    canonical_key: str
    start_offset: int
    end_offset: int
    provenance: str


@dataclass(frozen=True)
class StructuralReferenceResolutionResult:
    references: list[StructuralReferenceResolution]
    unique_keys: dict[str, StructuralReferenceResolution]
