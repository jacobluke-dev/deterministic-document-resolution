from __future__ import annotations

from observability.logger.levels import LogLevel
from observability.logger.message_logger import message_logger

from document_resolution.nlp.extraction.structural.types import (
    MATCH_STRATEGY,
    StructuralAnchor,
    StructuralReferenceEntry,
    StructuralReferenceLink,
)
from document_resolution.wiring.observability import sink


def _strength_and_strategy_for_tier(tier: str) -> tuple[float, MATCH_STRATEGY]:
    """Return the ordinal link-strength score and strategy for a match tier.

    The score is deterministic and explainable, not probabilistic. It encodes
    the relative quality of the positional rule that selected the anchor:

    - ``forward``: strongest positional support
    - ``backward``: exact-key match with weaker positional support
    - ``overlap``: weakest resolved case; chosen by fallback ordering
    - unresolved: no anchor target

    The values ``1.0 > 0.75 > 0.5 > 0.0`` are intentionally simple and spaced
    to distinguish strong, moderate, weak, and unresolved outcomes without
    implying statistical calibration.

    Args:
        tier: Positional match tier returned by anchor selection.

    Returns:
        Tuple of ``(strength, strategy)`` for the supplied tier.
    """
    if tier == "forward":
        return 1.0, "forward"
    if tier == "backward":
        return 0.75, "backward"
    if tier == "overlap":
        return 0.5, "overlap"
    return 0.0, "unresolved"


def _select_best_anchor(
    *,
    ref: StructuralReferenceEntry,
    candidates: list[StructuralAnchor],
) -> tuple[StructuralAnchor, str]:
    """Select the best anchor deterministically for a structural reference.

    Args:
        ref: Structural reference entry being linked.
        candidates: Candidate anchors sharing the same lookup key.

    Returns:
        Tuple of:
            - the selected structural anchor
            - the deterministic match tier used to select it

    Raises:
        ValueError: If ``candidates`` is empty.
    """
    forward = [a for a in candidates if a.start_offset >= ref.end_offset]
    if forward:
        return (
            min(
                forward,
                key=lambda a: (a.start_offset - ref.end_offset, a.ordinal),
            ),
            "forward",
        )

    backward = [a for a in candidates if a.end_offset <= ref.start_offset]
    if backward:
        return (
            min(
                backward,
                key=lambda a: (ref.start_offset - a.end_offset, a.ordinal),
            ),
            "backward",
        )

    return min(candidates, key=lambda a: a.ordinal), "overlap"


def build_structural_reference_links(
    *,
    references: list[StructuralReferenceEntry],
    anchor_index: dict[str, list[StructuralAnchor]],
) -> list[StructuralReferenceLink]:
    """Link structural reference entries to anchor spans deterministically.

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
                    match_strategy="unresolved",
                    strength=0.0,
                    provenance=ref.provenance,
                )
            )
            continue

        anchor, match_tier = _select_best_anchor(ref=ref, candidates=candidates)

        if match_tier not in {"forward", "backward", "overlap"}:
            message_logger(
                "structural.link.unsupported_match_tier",
                level=LogLevel.WARNING,
                logger_type="nlp.extraction",
                details={
                    "tier": match_tier,
                    "canonical_key": ref.canonical_key,
                    "kind": ref.kind,
                    "reference_span": (ref.start_offset, ref.end_offset),
                    "candidate_count": len(candidates),
                },
                db_sink=sink,
            )
        strength, strategy = _strength_and_strategy_for_tier(match_tier)

        out.append(
            StructuralReferenceLink(
                kind=ref.kind,
                label=ref.label,
                canonical_label=ref.canonical_label,
                normalized_key=ref.normalized_key,
                canonical_key=ref.canonical_key,
                reference_span=(ref.start_offset, ref.end_offset),
                target_span=(anchor.start_offset, anchor.end_offset),
                match_strategy=strategy,
                strength=strength,
                provenance=ref.provenance,
            )
        )

    return out
