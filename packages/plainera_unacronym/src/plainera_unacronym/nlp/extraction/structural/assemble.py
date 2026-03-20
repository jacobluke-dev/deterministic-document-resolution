from plainera_unacronym.nlp.extraction.structural.state import StructuralFlowState
from plainera_unacronym.nlp.extraction.structural.types import (
    StructuralReferenceEntry,
    StructuralReferenceResolutionResult,
)


def assemble_structural_reference_resolution_result(
    s: StructuralFlowState,
) -> StructuralReferenceResolutionResult:
    """Assemble final structural-reference resolution output from flow state.

    Builds the final result using canonicalised structural reference entries
    and resolved structural links stored on the flow state. The full ordered
    list of entries is preserved, and a canonical-key index is built for quick
    lookup of the first entry encountered for each canonical key.

    Args:
        s: Structural flow state containing structural reference entries and
            resolved structural links ready for final assembly.

    Returns:
        A ``StructuralReferenceResolutionResult`` containing:
            - ``references``: all structural reference entries in source order.
            - ``links``: all structural reference links in source order.
            - ``unique_keys``: mapping from canonical key to the first
              corresponding structural reference entry.
    """
    unique_keys: dict[str, StructuralReferenceEntry] = {}
    for ref in s.resolution_entries:
        unique_keys.setdefault(ref.canonical_key, ref)

    return StructuralReferenceResolutionResult(
        references=list(s.resolution_entries),
        links=list(s.link_entries),
        unique_keys=unique_keys,
    )
