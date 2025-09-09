import re
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    min_len: int = 2
    min_confidence_default: float = 0.50
    min_confidence_by_len: dict[int, float] = field(
        default_factory=lambda: {2: 0.72, 3: 0.60, 4: 0.55}
    )

    non_acronym_upper: frozenset[str] = frozenset({
        "OK",
        "PM",
        "MR", "MRS", "MS", "DR", "JR", "SR",
        "LTD", "PLC", "LLP", "LLC", "INC",
        "YES", "NO", "ON", "OFF"
    })

    max_len: int = 10
    # Allowed internal punctuation in acronyms (normalized for keying).
    allow_chars: str = "&/'’-"
    # Very small, locale-aware blacklist. Configurable/overrideable.
    soft_blacklist: frozenset[str] = frozenset({
        "AS", "AT", "BE", "BY", "DO", "GO", "IF", "IN", "IS", "OF", "ON", "OR", "SO", "TO", "AN"
    })

    locale: str = "en_GB"
    window_chars: int = 80
    # Letters only: ratio of uppercase letters over letters (digits ignored).
    require_caps_ratio: float = 0.7


@dataclass(frozen=True, slots=True)
class Occurrence:
    acronym: str                 # surface form as detected (not lowercased)
    start_offset: int
    end_offset: int              # end-exclusive
    confidence: float
    context_window: tuple[int, int]


@dataclass(frozen=True, slots=True)
class FirstOccurrence:
    acronym: str
    start_offset: int
    end_offset: int
    confidence: float


@dataclass(frozen=True, slots=True)
class DetectorResult:
    unique_acronyms: dict[str, FirstOccurrence]   # key = normalized_key
    occurrences: list[Occurrence]


pattern_cache: dict[tuple[int, int, str], re.Pattern[str]] = {}
