from __future__ import annotations

from dataclasses import replace

from plainera_unacronym.nlp.extraction.structural.types import (
    StructuralReferenceEntry,
    StructuralReferenceLink,
    StructuralReferenceResolutionResult,
)


def _shift_span(span: tuple[int, int], delta: int) -> tuple[int, int]:
    """Shift a structural span by a character offset.

    Args:
        span: Source span as ``(start, end)``.
        delta: Character offset to add to both positions.

    Returns:
        The original span when ``delta == 0``; otherwise a shifted span.
    """
    if delta == 0:
        return span
    return int(span[0]) + delta, int(span[1]) + delta


def shift_structural_reference_result(
    result: StructuralReferenceResolutionResult,
    delta: int,
) -> StructuralReferenceResolutionResult:
    """Shift all structural reference offsets into document coordinates.

    This is used when structural-reference results were produced from a text
    chunk and need to be projected back into full-document offsets.

    Args:
        result: Chunk-local structural reference result.
        delta: Character offset to apply to reference and target spans.

    Returns:
        The original result when ``delta == 0``; otherwise a new result with
        shifted references, links, and unique-key maps.
    """
    if delta == 0:
        return result

    references = [
        replace(
            ref,
            start_offset=int(ref.start_offset) + delta,
            end_offset=int(ref.end_offset) + delta,
        )
        for ref in result.references
    ]

    links = [
        replace(
            link,
            reference_span=_shift_span(link.reference_span, delta),
            target_span=None if link.target_span is None else _shift_span(link.target_span, delta),
        )
        for link in result.links
    ]

    unique_keys = {
        key: replace(
            ref,
            start_offset=int(ref.start_offset) + delta,
            end_offset=int(ref.end_offset) + delta,
        )
        for key, ref in result.unique_keys.items()
    }

    unique_links = {
        key: replace(
            link,
            reference_span=_shift_span(link.reference_span, delta),
            target_span=None if link.target_span is None else _shift_span(link.target_span, delta),
        )
        for key, link in result.unique_links.items()
    }

    return StructuralReferenceResolutionResult(
        references=references,
        links=links,
        unique_keys=unique_keys,
        unique_links=unique_links,
    )


def _reference_key(ref: StructuralReferenceEntry) -> tuple[object, ...]:
    return (
        ref.kind,
        ref.label,
        ref.canonical_label,
        ref.normalized_key,
        ref.canonical_key,
        int(ref.start_offset),
        int(ref.end_offset),
        ref.provenance,
    )


def _link_key(link: StructuralReferenceLink) -> tuple[object, ...]:
    target = None
    if link.target_span is not None:
        target = (int(link.target_span[0]), int(link.target_span[1]))

    return (
        link.kind,
        link.label,
        link.canonical_label,
        link.normalized_key,
        link.canonical_key,
        int(link.reference_span[0]), # start
        int(link.reference_span[1]), # end
        target,
        link.match_strategy,
        float(link.strength),
        link.provenance,
    )


def _is_resolved(link: StructuralReferenceLink) -> bool:
    return link.target_span is not None and link.match_strategy != "unresolved"


def _choose_unique_link(
    current: StructuralReferenceLink | None,
    candidate: StructuralReferenceLink,
) -> StructuralReferenceLink:
    if current is None:
        return candidate

    current_resolved = _is_resolved(current)
    candidate_resolved = _is_resolved(candidate)

    if not current_resolved and candidate_resolved:
        return candidate

    return current


def merge_structural_reference_results(
    chunk_results: list[tuple[int, StructuralReferenceResolutionResult]],
) -> StructuralReferenceResolutionResult:
    """Merge chunked structural-reference results into one document-level result.

    The merge process:
    1. Shifts chunk-local offsets into document coordinates.
    2. Deduplicates references and links by stable structural identity.
    3. Sorts merged references and links deterministically by span.
    4. Rebuilds ``unique_keys`` from first-seen references per canonical key.
    5. Rebuilds ``unique_links`` from merged links, preferring a resolved link
       over an unresolved one for the same canonical key.

    Args:
        chunk_results: Pairs of ``(delta, result)``, where ``delta`` is the
            chunk start offset in document coordinates.

    Returns:
        A merged, deterministically ordered structural-reference result.
    """
    shifted_results = [
        shift_structural_reference_result(result, delta)
        for delta, result in chunk_results
    ]

    reference_seen: set[tuple[object, ...]] = set()
    link_seen: set[tuple[object, ...]] = set()

    references: list[StructuralReferenceEntry] = []
    links: list[StructuralReferenceLink] = []

    for result in shifted_results:
        for ref in result.references:
            key = _reference_key(ref)
            if key in reference_seen:
                continue
            reference_seen.add(key)
            references.append(ref)

        for link in result.links:
            key = _link_key(link)
            if key in link_seen:
                continue
            link_seen.add(key)
            links.append(link)

    references.sort(key=lambda ref: (int(ref.start_offset), int(ref.end_offset), ref.canonical_key))
    links.sort(
        key=lambda link: (
            int(link.reference_span[0]), # start
            int(link.reference_span[1]), # end
            link.canonical_key,
        )
    )

    unique_keys: dict[str, StructuralReferenceEntry] = {}
    for ref in references:
        unique_keys.setdefault(ref.canonical_key, ref)

    unique_links: dict[str, StructuralReferenceLink] = {}
    for link in links:
        unique_links[link.canonical_key] = _choose_unique_link(
            unique_links.get(link.canonical_key),
            link,
        )

    return StructuralReferenceResolutionResult(
        references=references,
        links=links,
        unique_keys=unique_keys,
        unique_links=unique_links,
    )
