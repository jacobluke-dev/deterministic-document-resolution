from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StructuralReference:
    kind: str
    label: str
    start_offset: int
    end_offset: int
    normalized_key: str
    provenance: str


@dataclass(frozen=True)
class StructuralReferenceDetectorResult:
    references: list[StructuralReference]
