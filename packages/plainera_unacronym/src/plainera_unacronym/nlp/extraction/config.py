from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping

from plainera_unacronym.nlp.common.constants_regex import BRIDGES_DEFAULT, DEFAULT_STOPWORDS, INLINE_CUE_FRAGMENTS


@dataclass(frozen=True, slots=True)
class ConfidenceConfig:
    base_by_source: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType(
            {
                "first_occurrence_anchored": 0.85,  # first-occurrence local window + anchored regex patterns
                "parenthetical": 0.95,  # (ACR) long form (long form) ACR
                "inline": 0.8,  # cue-based: "ACR stands for Long Form" / "Long Form, abbreviated as ACR"
                "sentence_backref": 0.60,  # definition in earlier sentence(s), acronym later
                "all_occ_scan_parenthetical": 0.80,  # scan around all occrs, looser than anchored-first aka 'harvest'
            }
        )
    )

    # --- backref shaping ---
    backref_definitionish_boost: float = 0.10  # if we got a definition-ish span
    backref_initials_boost: float = 0.00  # if we fell back to initials span
    backref_lookback_penalty: float = 0.05  # per sentence beyond the nearest
    backref_distance_penalty_per_char: float = 0.0005  # per char (same shape as anchored)
    backref_distance_penalty_cap_chars: int = 200  # cap the distance penalty
    backref_uppercase_acronym_boost: float = 0.05  # if FO acronym token is ALL CAPS

    # Titlecase heuristic
    backref_titlecase_ratio_threshold: float = 0.80  # ratio of tokens starting uppercase
    backref_titlecase_boost: float = 0.05

    # disambiguation blend
    dist_weight: float = 0.75
    overlap_weight: float = 0.25
    # optional prior (keep small; tie-breaker only is also fine)
    sense_prior_weight: float = 0.0  # start 0.0; can raise later


@dataclass(frozen=True, slots=True)
class Tier2Config:
    mode: Literal["off", "auto", "on"] = "auto"
    model_name: str = "all-MiniLM-L6-v2"
    max_candidates: int = 6
    weight: float = 0.35

    # If True, Tier-2 only runs when Tier-1 chose None.
    # If False, Tier-2 may rerank even when Tier-1 chose, but you still gate via ceilings below.
    only_when_undecided: bool = False

    context_window_chars: int | None = 160  # if None, uses disambig.window_chars

    # AUTO eligibility gate:
    # - Tier-1 still "decides" at margin_threshold (e.g. 0.20)
    # - Tier-2 runs in auto-mode when r1.margin < auto_margin_ceiling (e.g. 0.75 for auditing)
    auto_margin_ceiling: float = 0.75  # eligibility in auto
    select_margin_threshold: float = 0.10  # acceptance when using blended scores (auto)


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    tier2: Tier2Config = field(default_factory=Tier2Config)

    # Phrase limits / toggles
    max_phrase_chars: int = 200
    stop: frozenset[str] = frozenset(DEFAULT_STOPWORDS)
    bridges: frozenset[str] = frozenset(BRIDGES_DEFAULT)

    # Acronym policy
    min_acr_len: int = 2

    # Inline cue regex fragments (case-insensitive)
    inline_cues: tuple[str, ...] = INLINE_CUE_FRAGMENTS

    sentence_backref_lookback: int = 2

    # Optional stricter gating
    require_two_words: bool = True

    # Optional plugin names (must be registered in plugins.registry)
    plugins: tuple[str, ...] = ()
