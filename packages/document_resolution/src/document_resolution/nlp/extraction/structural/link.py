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
    """Return a deterministic link-quality score for the selected anchor tier.

    This score is intentionally heuristic and ordinal rather than probabilistic.
    It does not estimate the true likelihood that a link is correct. Instead, it
    exposes a stable and explainable quality signal derived from the positional
    rule used to choose the anchor.

    Reasoning behind the tiers:

    * ``1.0`` for ``"forward"``
      A forward match is the strongest structural signal in the current model.
      When a reference appears before a matching heading and the heading starts
      at or after the reference end offset, this most closely reflects the
      common document pattern where narrative text points ahead to a later
      section, schedule, clause, or appendix. Because this is both exact-key and
      strongly positionally supported, it receives the highest score.

    * ``0.75`` for ``"backward"``
      A backward match is still exact-key and deterministic, but carries weaker
      positional support than a forward match. It indicates that no suitable
      forward anchor was found and the best match was instead a prior heading
      ending before the reference start offset. This is often still correct,
      especially in documents that refer back to previously introduced
      structures, but it is a weaker signal than a forward reference and is
      therefore scored lower.

    * ``0.5`` for ``"overlap"``
      An overlap / ordinal fallback occurs when neither a clean forward nor
      backward positional relationship exists and the tie is resolved by
      choosing the earliest anchor by ordinal. This still preserves
      determinism, but it is the weakest resolved case because the positional
      evidence is limited. A common example is when a structural heading is also
      detected as a structural reference inside its own anchor span. This is
      still useful to surface as resolved, but it should be distinguishable from
      stronger forward and backward matches.

    * ``0.0`` for unresolved links
      Unresolved links have no anchor target and therefore no positive link
      quality signal. They remain fully traceable in output, but carry the
      lowest possible score.

    Why these exact numeric values:
        The chosen values are deliberately spaced and easy to interpret:
        ``1.0 > 0.75 > 0.5 > 0.0``. They encode an ordered quality ladder
        without implying statistical calibration or false precision. The gaps
        are wide enough for downstream consumers to distinguish strong,
        moderate, weak, and unresolved outcomes, while remaining simple to
        reason about in tests and documentation.

    Args:
        tier: Deterministic positional match tier returned by anchor selection.

    Returns:
        Deterministic link-quality score for the tier.

    Raises:
        ValueError: If ``tier`` is unsupported.
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

    References are matched against anchors using the reference canonical key.
    When no anchor exists for a key, the reference remains unresolved with
    ``target_span=None`` and ``strength=0.0``. When one or more anchors are
    available, the best anchor is selected deterministically using
    ``_select_best_anchor``, and a deterministic graded link-quality score is
    assigned from the selected positional tier.

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
