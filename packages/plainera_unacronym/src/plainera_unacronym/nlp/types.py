import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DetectorConfig:
    min_len: int = 2
    max_len: int = 10
    # Allowed internal punctuation in acronyms (normalized for keying).
    allow_chars: str = "&/'’-"
    # Very small, locale-aware blacklist. Configurable/overrideable.
    blacklist: frozenset[str] = frozenset({"AM", "OK", "NO", "IT"})
    locale: str = "en_GB"
    window_chars: int = 80
    # Letters only: ratio of uppercase letters over letters (digits ignored).
    require_caps_ratio: float = 0.7


@dataclass(frozen=True)
class Occurrence:
    acronym: str                 # surface form as detected (not lowercased)
    start_offset: int
    end_offset: int              # end-exclusive
    confidence: float
    context_window: tuple[int, int]


@dataclass(frozen=True)
class FirstOccurrence:
    acronym: str
    start_offset: int
    end_offset: int
    confidence: float


@dataclass(frozen=True)
class DetectorResult:
    unique_acronyms: dict[str, FirstOccurrence]   # key = normalized_key
    occurrences: list[Occurrence]


pattern_cache: dict[tuple[int, int, str], re.Pattern[str]] = {}
