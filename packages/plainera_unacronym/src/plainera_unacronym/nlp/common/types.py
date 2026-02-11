import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cached_property
from types import MappingProxyType
from typing import Any, FrozenSet, Literal, Mapping, Optional, TypeAlias, cast

from plainera_unacronym.nlp.common.constants_regex import ALLOW_CHARS, DottedMode

SCHEMA_VERSION = "1.1.0"


# -------------------------- SPANS ------------------------------------

Span: TypeAlias = tuple[int, int]
TextSpanTuple: TypeAlias = tuple[str, int, int]


@dataclass(frozen=True, slots=True)
class TextSpan:
    text: str
    start: int
    end: int  # half-open [start, end)

    @property
    def span(self) -> Span:
        return self.start, self.end

    @property
    def length(self) -> int:
        return self.end - self.start


# -------------------------- STRATEGIES ------------------------------------

# TODO these will be adjusted to tier 1 tier 2 or some other consideration right now these will do
Extraction_strategy: TypeAlias = Literal[
    "anchored+harvest",
    "hybrid-filled",
    "global",
    "anchored+harvest+global",
]

Definition_strategy = Literal[
    "direct_def",
    "helper_def_before",
    "helper_def_after",
    "helper_inline_after",
]


# -------------------------- OCCURRENCE -------------------------------------


@dataclass(frozen=True, slots=True)
class Occurrence:
    acronym: str  # surface form as detected (not lowercased)
    start_offset: int
    end_offset: int  # end-exclusive
    occurrence_confidence: float
    context_window: Span  # (left_idx, right_idx) in the original text
    normalized_key: str | None = None
    reasons: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class FirstOccurrence:
    acronym: str
    start_offset: int
    end_offset: int
    occurrence_confidence: float
    normalized_key: str | None = None


@dataclass
class AcronymSense:
    acronym: str
    definition: str  # tightened, normalized label ("European Medicines Agency")
    sense_id: str  # stable key, e.g., "ema|european_medicines_agency"
    sense_confidence: float
    def_spans: list[Span]  # locations where this sense was defined
    support: int  # number of defining mentions merged into this sense


@dataclass
class OccurrenceLite:
    acronym: str
    start: int
    end: int


@dataclass
class OccurrenceResolution:
    acronym: str
    start: int
    end: int
    chosen_sense_id: Optional[str]  # None if ambiguous
    candidates: dict[str, float]  # sense_id -> score (0..1)
    margin: float  # top - second best


# -------------------------- DETECTION -------------------------------------


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    min_len: int = 2
    min_confidence_default: float = 0.50
    min_confidence_by_len: Mapping[int, float] = field(
        default_factory=lambda: MappingProxyType({2: 0.72, 3: 0.60, 4: 0.55})
    )

    non_acronym_upper: frozenset[str] = frozenset(
        {"OK", "PM", "MR", "MRS", "MS", "DR", "JR", "SR", "LTD", "PLC", "LLP", "LLC", "INC", "YES", "NO", "ON", "OFF"}
    )
    whitelist_two_letter: frozenset[str] = frozenset({"US", "UK", "EU", "UN"})

    max_len: int = 10
    # Allowed internal punctuation in acronyms (normalized for keying).
    allow_chars: str = ALLOW_CHARS
    # Very small, locale-aware blacklist. Configurable/overrideable.
    blacklist: frozenset[str] = frozenset(
        {"AS", "AT", "BE", "BY", "DO", "GO", "IF", "IN", "IS", "OF", "ON", "OR", "SO", "TO", "AN"}
    )
    user_org_blacklist: frozenset[str] = frozenset()

    locale: str = "en_GB"
    window_chars: int = 140
    # Letters only: ratio of uppercase letters over letters (digits ignored).
    require_caps_ratio: float = 0.7
    enable_dotted: bool = False
    debug_reasons: bool = False
    enable_mixed_case: bool = True
    dotted_display: DottedMode = "strip"  # "strip" | "preserve"
    require_caps_ratio_mixed: float = 0.5
    enabled_domains: FrozenSet[str] = frozenset()
    domain_cfg: Mapping[str, Any] = field(default_factory=dict)
    debug_anomalies: bool = False  # set to true if we want to run  message logger in dev / live envs

    @cached_property
    def allow_chars_set(self) -> frozenset[str]:
        return frozenset(self.allow_chars)


@dataclass(frozen=True, slots=True)
class DetectorResult:
    unique_acronyms: dict[str, FirstOccurrence]  # key = normalized_key
    occurrences: list[Occurrence]


# -------------------------- EXTRACTION -------------------------------------


@dataclass(frozen=True, slots=True)
class ExtractedDefinition:
    acronym: str
    definition: str  # normalized
    source: str  # "all_occ_scan_parenthetical"
    definition_confidence: float
    acr_start: int
    acr_end: int
    def_start: int
    def_end: int
    original_definition: str
    kind: str = "unknown"
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class InTextPick:
    definition: str
    acr_span: Span
    def_span: Span
    definition_confidence: float
    original_definition: str
    kind: str = "unknown"
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    # map normalized_key -> pick (nearest in-text definition) or None if not found
    picks: dict[str, Optional[InTextPick]]
    # all definition locations considered (anchored-window matches if no global run,
    # or full global matches if we did the fallback)
    definitions: list[ExtractedDefinition]
    # which strategy ultimately produced 'picks' / 'definitions'
    extraction_strategy: Extraction_strategy
    # convenience metric: fraction of acronyms with a pick
    coverage: float
    # normalized keys that had no in-text definition
    missing_keys: tuple[str, ...]

    senses_by_acronym: dict[str, list[AcronymSense]] = field(default_factory=dict)
    sense_index: dict[str, AcronymSense] = field(default_factory=dict)  # sense_id -> sense
    resolutions: list[OccurrenceResolution] = field(default_factory=list)
    ambiguous_keys: tuple[str, ...] = field(default_factory=tuple)  # acronyms with >1 senses
    undecided: list[OccurrenceResolution] = field(default_factory=list)  # chosen_sense_id is None


class OccurrenceBuildError(Exception):
    pass


# -------------------------- MISC -------------------------------------

pattern_cache: dict[tuple[Any, ...], re.Pattern[str]] = {}

soft_dotted_drop: frozenset[str] = frozenset({"EG", "IE", "AKA"})

INLINE = "inline"
INLINE_BEFORE = "inline_before"

INLINE_KINDS = {INLINE, INLINE_BEFORE}


JsonDict = dict[str, Any]


# ---------------------  helpers ----------------------


def as_str_set(x: Any, *, default: Iterable[str]) -> set[str]:
    """Coerce a config-provided stop/bridge collection into a concrete set[str]."""
    if x is None:
        return set(default)
    # If someone passes a single string, treat it as one token, not chars.
    if isinstance(x, str):
        return {x}
    return set(cast(Iterable[str], x))


ExtractionStrategy: TypeAlias = Literal[
    "anchored+harvest",
    "hybrid-filled",
    "global",
    "anchored+harvest+global",
]


def _compute_strategy(
    *, has_gapfill: bool, has_global: bool, has_anchored: bool, has_harvest: bool
) -> ExtractionStrategy:
    if has_gapfill:
        return "hybrid-filled"
    if has_global:
        # keep this strict so it matches your alias exactly
        return "anchored+harvest+global" if (has_anchored and has_harvest) else "global"
    return "anchored+harvest"
