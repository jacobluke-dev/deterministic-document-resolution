from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DefinedTermSense:
    term: str
    start_offset: int
    end_offset: int
    normalized_key: str
    sense_id: str
    provenance: str


@dataclass(frozen=True)
class DefinedTermOccurrence:
    term: str
    start_offset: int
    end_offset: int
    normalized_key: str
    occurrence_confidence: float = 1.0
    segment_window: Optional[str] = None


@dataclass(frozen=True)
class DefinedTermDetectorResult:
    occurrences: list[DefinedTermOccurrence]
    unique_terms: dict[str, DefinedTermSense]
