from __future__ import annotations

from dataclasses import dataclass, field

from document_resolution.nlp.extraction.tiers.config import ResolutionConfig, Tier2Config


@dataclass(frozen=True, slots=True)
class DefinedTermExtractionConfig:
    """Configuration for defined-term extraction and resolution.

    Controls the deterministic Tier-1 resolver, optional Tier-2 semantic rerank,
    and the tie-break policy used when multiple candidate meanings for the same
    defined term are not meaningfully separable.

    Attributes:
        tier2: Configuration for optional Tier-2 semantic reranking.
        multi_tier: Shared multi-tier resolution configuration.
        tier_1_window_chars: Character window used to derive local occurrence
            context for Tier-1 lexical-overlap scoring when no precomputed
            segment window is available.
        tier_1_margin_threshold: Minimum normalised score margin required for a
            clear Tier-1 winner. Candidates within this margin are treated as
            not meaningfully separable.
        lexical_overlap_weight: Weight applied to lexical overlap between the
            occurrence context and candidate definition text.
        section_proximity_weight: Weight applied to structural section-path
            proximity between an occurrence and a candidate meaning.
        directionality_weight: Weight applied to whether a candidate definition
            was introduced before or after the occurrence being resolved.
        intro_type_weight: Weight applied to the heuristic strength of the
            candidate introduction form, for example ``quoted_means`` versus
            ``parenthetical_alias``.
        prefer_prior_definitions: When enabled, prefer the earliest introduced
            candidate by document order in Tier-1 tie or near-tie scenarios
            where candidates are otherwise not separable above the configured
            margin threshold. Strong non-tied winners are unchanged.
    """

    tier2: Tier2Config = field(default_factory=Tier2Config)
    multi_tier: ResolutionConfig = field(default_factory=ResolutionConfig)

    tier_1_window_chars: int = 180
    tier_1_margin_threshold: float = 0.20

    lexical_overlap_weight: float = 1.0
    section_proximity_weight: float = 1.0
    directionality_weight: float = 1.0
    intro_type_weight: float = 1.0

    prefer_prior_definitions: bool = True  # might stay might not, I'm not sure
