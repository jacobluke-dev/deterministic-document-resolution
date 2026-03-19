from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class StructuralReferenceDetectorConfig:
    """Created for symmetry """
    pass

@dataclass(frozen=True)
class StructuralReferenceExtractionConfig:
    """Configuration for structural-reference extraction."""

    convert_roman_numerals: bool = False
    preserve_source_case_in_labels: bool = True
