from __future__ import annotations

from plainera_unacronym.nlp.extraction.structural.types import (
    StructuralAnchor,
    StructuralReferenceEntry,
    StructuralReferenceLink,
)


def _select_best_anchor(
    *,
    ref: StructuralReferenceEntry,
    candidates: list[StructuralAnchor],
) -> StructuralAnchor:
    """Select the best anchor deterministically for a structural reference.

    Candidate anchors are chosen using structural proximity with a stable
    preference order:
      1. nearest anchor starting at or after the reference end offset,
      2. otherwise nearest anchor ending before the reference start offset,
      3. otherwise earliest anchor by ordinal.

    This keeps linking deterministic when multiple anchors share the same
    lookup key.

    Args:
        ref: Structural reference entry being linked.
        candidates: Candidate anchors sharing the same lookup key.

    Returns:
        The selected structural anchor.

    Raises:
        ValueError: If ``candidates`` is empty.
    """
    forward = [a for a in candidates if a.start_offset >= ref.end_offset]
    if forward:
        return min(
            forward,
            key=lambda a: (a.start_offset - ref.end_offset, a.ordinal),
        )

    backward = [a for a in candidates if a.end_offset <= ref.start_offset]
    if backward:
        return min(
            backward,
            key=lambda a: (ref.start_offset - a.end_offset, a.ordinal),
        )

    return min(candidates, key=lambda a: a.ordinal)


def build_structural_reference_links(
    *,
    references: list[StructuralReferenceEntry],
    anchor_index: dict[str, list[StructuralAnchor]],
) -> list[StructuralReferenceLink]:
    """Link structural reference entries to anchor spans deterministically.

    References are matched against anchors using the reference canonical key.
    When no anchor exists for a key, the reference remains unresolved with
    ``target_span=None`` and ``confidence=0.0``. When one or more anchors are
    available, the best anchor is selected deterministically using
    ``_select_best_anchor``.

    Args:
        references: Canonicalized structural reference entries in source order.
        anchor_index: Mapping from anchor lookup key to ordered structural
            anchors sharing that key.

    Returns:
        List of structural reference links in input order.
    """
    out: list[StructuralReferenceLink] = []

    for ref in references:
        candidates = anchor_index.get(ref.canonical_key, [])

        if not candidates:
            out.append(
                StructuralReferenceLink(
                    kind=ref.kind,
                    label=ref.label,
                    canonical_label=ref.canonical_label,
                    normalized_key=ref.normalized_key,
                    canonical_key=ref.canonical_key,
                    reference_span=(ref.start_offset, ref.end_offset),
                    target_span=None,
                    confidence=0.0,
                    provenance=ref.provenance,
                )
            )
            continue

        anchor = _select_best_anchor(ref=ref, candidates=candidates)

        out.append(
            StructuralReferenceLink(
                kind=ref.kind,
                label=ref.label,
                canonical_label=ref.canonical_label,
                normalized_key=ref.normalized_key,
                canonical_key=ref.canonical_key,
                reference_span=(ref.start_offset, ref.end_offset),
                target_span=(anchor.start_offset, anchor.end_offset),
                confidence=1.0,
                provenance=ref.provenance,
            )
        )

    return out
