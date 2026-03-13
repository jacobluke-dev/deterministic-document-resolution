from dataclasses import dataclass
from typing import Optional, Literal

IntroKind = Literal[
    "quoted_means",
    "quoted_shall_mean",
    "bare_means",
    "bare_shall_mean",
    "parenthetical_alias",
    "unknown",
]

@dataclass(frozen=True)
class DefinedTermIntroduction:
    term: str
    start_offset: int
    end_offset: int
    normalized_key: str
    provenance: str
    intro_kind: IntroKind = "unknown"


@dataclass(frozen=True)
class DefinedTermMention:
    term: str
    start_offset: int
    end_offset: int
    normalized_key: str
    confidence: float = 1.0
    segment_window: Optional[str] = None


@dataclass(frozen=True)
class DefinedTermDetectorResult:
    introductions: list[DefinedTermIntroduction]
    mentions: list[DefinedTermMention]
    unique_terms: dict[str, DefinedTermIntroduction]
