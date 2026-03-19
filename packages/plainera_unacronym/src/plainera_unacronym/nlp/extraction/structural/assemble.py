from plainera_unacronym.nlp.extraction.structural.state import StructuralFlowState
from plainera_unacronym.nlp.extraction.structural.types import (
    StructuralReferenceResolutionResult, StructuralReferenceResolution,
)


def assemble_structural_reference_resolution_result(
    s: StructuralFlowState,
) -> StructuralReferenceResolutionResult:
    """Assemble final structural-reference extraction output from flow state.

    Builds the final extraction result using the accumulated structural
    resolution entries stored on the flow state. The full ordered list of
    resolution entries is preserved, and a canonical-key index is built for
    quick lookup of the first resolution encountered for each canonical key.

    Args:
        s: Structural flow state containing resolved structural reference
            entries ready for final assembly.

    Returns:
        A ``StructuralReferenceResolutionResult`` containing:
            - ``references``: all structural resolution entries in source order.
            - ``unique_keys``: mapping from canonical key to the first
              corresponding structural resolution entry.
    """
    unique_keys: dict[str, StructuralReferenceResolution] = {}
    for ref in s.resolution_entries:
        unique_keys.setdefault(ref.canonical_key, ref)

    return StructuralReferenceResolutionResult(
        references=list(s.resolution_entries),
        unique_keys=unique_keys,
    )
