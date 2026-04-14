from plainera_unacronym.nlp.extraction.structural.state import StructuralFlowState
from plainera_unacronym.nlp.extraction.structural.types import (
    StructuralReferenceEntry,
    StructuralReferenceLink,
    StructuralReferenceResolutionResult,
)


def assemble_structural_reference_resolution_result(
    s: StructuralFlowState,
) -> StructuralReferenceResolutionResult:
    """Assemble final structural-reference resolution output from flow state.

    Builds the final result using canonicalised structural reference entries
    and resolved structural links stored on the flow state. The full ordered
    lists of entries and links are preserved, and canonical-key indexes are
    built for quick lookup of the first entry and first link encountered for
    each canonical key.

    Args:
        s: Structural flow state containing structural reference entries and
            resolved structural links ready for final assembly.

    Returns:
        A ``StructuralReferenceResolutionResult`` containing:
            - ``references``: all structural reference entries in source order.
            - ``links``: all structural reference links in source order.
            - ``unique_keys``: mapping from canonical key to the first
              corresponding structural reference entry.
            - ``unique_links``: mapping from canonical key to the first
              corresponding structural reference link.
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
