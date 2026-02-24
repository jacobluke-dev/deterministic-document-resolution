from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Mapping, TypeAlias, cast

from plainera_unacronym.nlp.common.constants_regex import ALLOW_CHARS, DottedMode

if TYPE_CHECKING:
    from plainera_unacronym.nlp.extraction.tiers.types import Tier2OccurrenceRanking, Tier2Report

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
    """
    Represents a single “meaning” (sense) of an acronym within a document.

    An `AcronymSense` is constructed from one or more extracted in-text definitions
    that normalise to the same `(acronym, definition)` identity (e.g. multiple
    mentions of “European Medicines Agency (EMA)” across the text). It is used as
    the unit of choice during occurrence-level disambiguation.

    Key ideas:
      - `sense_id` is a stable identifier (typically derived from the acronym plus a
        slug of the tightened definition) used for indexing and resolution outputs.
      - `def_spans` records where this sense was defined in the source text; these
        spans drive proximity-based scoring and distance tie-breaks.
      - `sense_confidence` is a deterministic strength signal for the sense (e.g. the
        max confidence among supporting definitions). It may be used as a small prior
        for near-tie breaking, but should not override structural validity gates.
      - `support` counts how many definition instances were merged into this sense.

    Attributes:
        acronym: Uppercased acronym string (e.g. "EMA").
        definition: Tightened/normalised definition label for this sense.
        sense_id: Stable key for this sense (e.g. "ema|european_medicines_agency").
        sense_confidence: Deterministic confidence scalar in [0, 1] for this sense.
        def_spans: List of (start, end) spans where this sense is defined in the text.
        support: Number of definition instances merged into this sense.
    """

    acronym: str
    definition: str  # tightened, normalized label ("European Medicines Agency")
    sense_id: str  # stable key, e.g., "ema|european_medicines_agency"
    sense_confidence: float
    def_spans: list[Span]  # locations where this sense was defined
    support: int  # number of defining mentions merged into this sense


@dataclass
class OccurrenceLite:
    """
    Minimal representation of an acronym occurrence in the source text.

    This type is intentionally lightweight: it captures only the acronym surface
    and its character offsets. It is fed into disambiguation, where each occurrence
    is scored against candidate `AcronymSense` objects using local context windows
    and distance to definition spans.

    Attributes:
        acronym: Acronym surface string as detected (typically uppercased upstream).
        start: Start character offset of the occurrence in the document.
        end: End character offset (exclusive) of the occurrence in the document.
    """

    acronym: str
    start: int
    end: int


@dataclass
class OccurrenceResolution:
    """
    Resolution result for a single acronym occurrence.

    Holds the chosen sense (or None) plus per-sense scores and the top-two score gap.

    Args:
        acronym: Acronym surface for this occurrence.
        start: Start offset (inclusive) in the source text.
        end: End offset (exclusive) in the source text.
        chosen_sense_id: Selected sense_id, or None if ambiguous.
        candidate_scores: Mapping of sense_id -> score in [0.0, 0.99].
        gap: Absolute gap (top_score - second_score), 0.0 if <2 candidates.
        margin:
    """

    acronym: str
    start: int
    end: int
    chosen_sense_id: str | None
    candidate_scores: dict[str, float]
    gap: float
    margin: float


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
    enabled_domains: frozenset[str] = frozenset()
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
    """
    Normalised definition evidence produced by an extraction strategy.
    Carries absolute spans into the source text plus provenance and confidence.
    This is the “ledger” of all candidates considered, not necessarily the final pick.
    Used for dedupe/merge, sense building, debugging, and traceability.

    Args:
        acronym: Acronym key/surface for this definition evidence.
        definition: Normalised/tightened definition text.
        source: Provenance label for where this evidence came from.
        definition_confidence: Relative strength score for ranking/selection.
        acr_start: Absolute start offset of the acronym span in the text.
        acr_end: Absolute end offset of the acronym span in the text.
        def_start: Absolute start offset of the definition span in the text.
        def_end: Absolute end offset of the definition span in the text.
        original_definition: Raw definition slice prior to normalisation.
        kind: Pattern/shape identifier (e.g. inline, def_before).
        reasons: Human-readable scoring/decision traces.
    """

    acronym: str
    definition: str  # normalized
    source: str
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
    """
    The chosen in-text definition for a single acronym key.
    Stores the “best” candidate as spans + canonical definition for downstream use.
    Typically selected from merged definition evidence, but may be filled heuristically.
    Intended for consumers that want one answer per acronym (plus confidence/reasons).

    Args:
        definition: Normalised/tightened definition text selected as the winner.
        acr_span: Absolute (start, end) offsets for the acronym occurrence.
        def_span: Absolute (start, end) offsets for the definition span.
        definition_confidence: Relative strength score for this pick.
        original_definition: Raw definition slice prior to normalisation.
        kind: Pattern/shape identifier describing how it was matched.
        route: Internal label for how the pick was chosen (may differ from source).
        reasons: Human-readable scoring/decision traces.
    """

    definition: str
    acr_span: Span
    def_span: Span
    definition_confidence: float
    original_definition: str
    kind: str = "unknown"
    route: str = "unknown"
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """
    Output bundle for Tier-1 extraction and selection.
    Provides per-acronym winners plus the full set of definition evidence observed.
    `picks` is the consumer-facing map; `definitions` is the evidence ledger.
    Includes coverage/missing metrics and optional sense-resolution artefacts.

    Args:
        picks: Normalised acronym key -> selected in-text pick or None.
        definitions: All extracted definition evidence from all strategies.
        coverage: Fraction of acronym keys with a non-null pick.
        missing_keys: Normalised keys with no pick after all selection steps.
        senses_by_acronym: Candidate senses grouped by acronym key.
        sense_index: Global sense lookup by sense_id.
        resolutions: Per-occurrence resolution decisions.
        ambiguous_keys: Keys with more than one viable sense.
        undecided: Resolutions where no sense could be chosen deterministically.
    """

    # map normalized_key -> pick (nearest in-text definition) or None if not found
    picks: dict[str, InTextPick | None]
    # all definition locations considered (anchored-window matches if no global run,
    # or full global matches if we did the fallback)
    definitions: list[ExtractedDefinition]
    # convenience metric: fraction of acronyms with a pick
    coverage: float
    # normalized keys that had no in-text definition
    missing_keys: tuple[str, ...]

    senses_by_acronym: dict[str, list[AcronymSense]] = field(default_factory=dict)
    sense_index: Mapping[str, AcronymSense] = field(default_factory=dict)  # sense_id -> sense
    resolutions: list[OccurrenceResolution] = field(default_factory=list)
    ambiguous_keys: tuple[str, ...] = field(default_factory=tuple)  # acronyms with >1 senses
    undecided: list[OccurrenceResolution] = field(default_factory=list)  # chosen_sense_id is None
    tier2_report: Tier2Report | None = None
    tier2_ranked: tuple[Tier2OccurrenceRanking, ...] = ()


class OccurrenceBuildError(Exception):
    pass


# -------------------------- MISC -------------------------------------

pattern_cache: dict[tuple[Any, ...], re.Pattern[str]] = {}

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
