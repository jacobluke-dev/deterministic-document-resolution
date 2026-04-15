from __future__ import annotations

from dataclasses import dataclass, field

from document_resolution.nlp.common.types import DefinedTermDetectorConfig
from document_resolution.nlp.detection.defined_terms import DefinedTermDetectorResult
from document_resolution.nlp.extraction.defined_terms.config import DefinedTermExtractionConfig
from document_resolution.nlp.extraction.defined_terms.structure import TermStructureIndex
from document_resolution.nlp.extraction.defined_terms.tiers.work import TermTier1Work, TermTier2Work
from document_resolution.nlp.extraction.defined_terms.types import (
    TermDefinitionEntry,
    TermMeaning,
    TermResolutionResult,
)


@dataclass
class TermFlowState:
    """Mutable state container for the defined-term resolution pipeline.

    This object is passed through the staged extraction flow and accumulates the
    intermediate artefacts produced at each step, including detector output,
    structural context, extracted definitions, Tier-1 and Tier-2 work products,
    and the final assembled resolution result.

    Attributes:
        text: Full source text being processed by the pipeline.
        det_cfg: Active defined-term detector configuration for this run.
        ext_cfg: Active defined-term extraction and resolution configuration for
            this run.
        last_info: Short human-readable status string describing the most recent
            completed stage.
        det_res: Detector output containing introductions, mentions, and
            unique-term mappings.
        term_meaning_index: Legacy or transitional top-level mapping from
            normalised term key to candidate meanings. Current Tier-1 work
            generally stores the active meaning index data.
        structure_index: Optional document structure index used for section-path
            aware scoring and extraction.
        definition_entries: Extracted definition entries derived from detected
            introductions.
        tier_1: Tier-1 working state containing candidate meanings, occurrences,
            and deterministic ranking outputs.
        tier_2: Tier-2 working state containing semantic rerank outputs and
            report metadata.
        extr: Final assembled defined-term resolution result, when available.
    """

    text: str
    det_cfg: DefinedTermDetectorConfig
    ext_cfg: DefinedTermExtractionConfig
    last_info: str = ""

    det_res: DefinedTermDetectorResult | None = None

    term_meaning_index: dict[str, tuple[TermMeaning, ...]] = field(default_factory=dict)
    structure_index: TermStructureIndex | None = None
    definition_entries: list[TermDefinitionEntry] = field(default_factory=list)

    tier_1: TermTier1Work = field(default_factory=TermTier1Work, repr=False)
    tier_2: TermTier2Work = field(default_factory=TermTier2Work, repr=False)

    extr: TermResolutionResult | None = None
