from document_resolution.nlp.extraction.structural.state import StructuralFlowState
from document_resolution.nlp.extraction.structural.types import (
    StructuralReferenceEntry,
    StructuralReferenceLink,
    StructuralReferenceResolutionResult,
)


def assemble_structural_reference_resolution_result(
    s: StructuralFlowState,
) -> StructuralReferenceResolutionResult:
    """Assemble final structural-reference resolution output from flow state.

    Args:
        s: Structural flow state containing structural reference entries and
            resolved structural links ready for final assembly.

    Returns:
        A ``StructuralReferenceResolutionResult``
    """
    unique_keys: dict[str, StructuralReferenceEntry] = {}
    for ref in s.reference_entries:
        unique_keys.setdefault(ref.canonical_key, ref)

    unique_links: dict[str, StructuralReferenceLink] = {}
    for link in s.link_entries:
        existing = unique_links.get(link.canonical_key)

        if existing is None:
            unique_links[link.canonical_key] = link
            continue

        existing_is_resolved = existing.target_span is not None
        candidate_is_resolved = link.target_span is not None

        if not existing_is_resolved and candidate_is_resolved:
            unique_links[link.canonical_key] = link

    return StructuralReferenceResolutionResult(
        references=list(s.reference_entries),
        links=list(s.link_entries),
        unique_keys=unique_keys,
        unique_links=unique_links,
    )
