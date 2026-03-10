from dataclasses import dataclass, field

from plainera_unacronym.nlp.extraction.config import Tier2Config, ResolutionConfig


@dataclass(frozen=True, slots=True)
class DefinedTermExtractionConfig:
    tier2: Tier2Config = field(default_factory=Tier2Config)
    multi_tier: ResolutionConfig = field(default_factory=ResolutionConfig)

    tier_1_window_chars: int = 180
    tier_1_margin_threshold: float = 0.20

    lexical_overlap_weight: float = 1.0
    section_proximity_weight: float = 1.0
    directionality_weight: float = 1.0
    intro_type_weight: float = 1.0

    prefer_prior_definitions: bool = True
