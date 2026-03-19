from __future__ import annotations

from plainera_unacronym.nlp.extraction.structural.state import StructuralFlowState
from plainera_unacronym.nlp.extraction.structural.types import (
    StructuralReferenceResolutionResult,
)


def assemble_structural_reference_resolution_result(
    s: StructuralFlowState,
) -> StructuralReferenceResolutionResult:
    """Assemble final structural-reference extraction output."""
    unique_keys = {}
    for ref in s.resolution_entries:
        unique_keys.setdefault(ref.canonical_key, ref)

    return StructuralReferenceResolutionResult(
        references=list(s.resolution_entries),
        unique_keys=unique_keys,
    )
