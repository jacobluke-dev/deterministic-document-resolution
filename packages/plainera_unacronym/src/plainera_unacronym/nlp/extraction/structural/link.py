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
    """Choose the best anchor deterministically for a structural reference.

    Preference order:
      1. nearest anchor starting at or after the reference end
      2. otherwise nearest anchor ending before the reference start
      3. otherwise first anchor by ordinal
    """
    forward = [a for a in candidates if a.start_offset >= ref.end_offset]
    if forward:
        return min(forward, key=lambda a: (a.start_offset - ref.end_offset, a.ordinal))

    backward = [a for a in candidates if a.end_offset <= ref.start_offset]
    if backward:
        return min(backward, key=lambda a: (ref.start_offset - a.end_offset, a.ordinal))

    return min(candidates, key=lambda a: a.ordinal)


def build_structural_reference_links(
    *,
    references: list[StructuralReferenceEntry],
    anchor_index: dict[str, list[StructuralAnchor]],
) -> list[StructuralReferenceLink]:
    """Resolve structural reference entries to anchor spans deterministically."""
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
