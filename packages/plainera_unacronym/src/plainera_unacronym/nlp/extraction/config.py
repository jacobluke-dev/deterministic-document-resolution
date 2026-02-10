from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from plainera_unacronym.nlp.common.constants_regex import BRIDGES_DEFAULT, DEFAULT_STOPWORDS, INLINE_CUE_FRAGMENTS


@dataclass(frozen=True, slots=True)
class ConfidenceConfig:
    base_by_source: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({
            "anchored": 0.85, # first-occurrence local window + anchored regex patterns RENAME this 'first_occurrence_anchored'
            "parenthetical": 0.95, # (ACR) long form (long form) ACR
            "backref": 0.60, # definition in earlier sentence(s), acronym later
            "all_occ_scan_parenthetical": 0.80, # scan around all occurrences, looser than anchored-first aka 'harvest'
        })
    )

    # disambiguation blend
    dist_weight: float = 0.75
    overlap_weight: float = 0.25
    # optional prior (keep small; tie-breaker only is also fine)
    sense_prior_weight: float = 0.0  # start 0.0; can raise later


@dataclass(frozen=True, slots=True)
class ExtractionConfig:

    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    # parenthetical_allows: tuple[Callable[[str, str], bool], ...] = ()
    # Phrase limits / toggles
    max_phrase_chars: int = 200
    enabled_parenthetical: bool = True
    enabled_inline: bool = True
    stop: frozenset[str] = frozenset(DEFAULT_STOPWORDS)
    bridges: frozenset[str] = frozenset(BRIDGES_DEFAULT)

    # Acronym policy
    min_acr_len: int = 2
    # max_acr_len: int = 10
    # acr_allowed: str = r"A-Z0-9&./-"

    # Confidence weights
    conf_parenthetical: float = 0.95
    conf_inline: float = 0.80

    # Inline cue regex fragments (case-insensitive)
    inline_cues: tuple[str, ...] = INLINE_CUE_FRAGMENTS

    sentence_backref_lookback: int = 2 # only used in tests

    # Optional stricter gating
    require_two_words: bool = True

    # Optional plugin names (must be registered in plugins.registry)
    plugins: tuple[str, ...] = ()

    # window_chars: int = 320
    # margin_threshold: float = 0.20
