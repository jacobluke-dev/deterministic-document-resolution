from __future__ import annotations

from plainera_unacronym.nlp.detection.structural.types import StructuralReference
from plainera_unacronym.nlp.extraction.structural.config import (
    StructuralReferenceExtractionConfig,
)
from plainera_unacronym.nlp.extraction.structural.types import (
    StructuralReferenceResolution,
)


def build_structural_reference_resolutions(
    *,
    references: list[StructuralReference],
    cfg: StructuralReferenceExtractionConfig,
) -> list[StructuralReferenceResolution]:
    """Build structural-reference resolution entries from detected references."""
    out: list[StructuralReferenceResolution] = []

    for ref in references:
        canonical_label = ref.label
        canonical_key = ref.normalized_key

        # UN-92 slot:
        # if cfg.convert_roman_numerals:
        #     canonical_label = ...
        #     canonical_key = ...

        out.append(
            StructuralReferenceResolution(
                kind=ref.kind,
                label=ref.label,
                canonical_label=canonical_label,
                normalized_key=ref.normalized_key,
                canonical_key=canonical_key,
                start_offset=ref.start_offset,
                end_offset=ref.end_offset,
                provenance=ref.provenance,
            )
        )

    return out
