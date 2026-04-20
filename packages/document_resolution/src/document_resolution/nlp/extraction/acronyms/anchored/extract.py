from collections.abc import Mapping
from typing import Optional

from document_resolution.nlp.common.types import ExtractedDefinition, FirstOccurrence, InTextPick, Span
from document_resolution.nlp.extraction.acronyms.anchored.clean import clean_definition
from document_resolution.nlp.extraction.acronyms.anchored.patterns import compile_anchored_for_surface
from document_resolution.nlp.extraction.acronyms.anchored.spans import resolve_def_span
from document_resolution.nlp.extraction.acronyms.config import ExtractionConfig
from document_resolution.nlp.extraction.acronyms.engine.confidence import base_for_kind


def _build_local_window(
    text: str,
    fo: FirstOccurrence,
    window_left: int,
    window_right: int,
) -> tuple[int, int, str]:
    """Build a bounded local text window around a first occurrence.

    Args:
        text (str): Full document text.
        fo (FirstOccurrence): First occurrence acr.
        window_left (int): No. chars to the left of `fo.start_offset`.
        window_right (int): No. chars to the right of `fo.end_offset`.

    Returns:
        tuple[int, int, str]: `(left, right, seg)` where:
            - `left` is the clamped start index (inclusive),
            - `right` is the clamped end index (exclusive),
            - `seg` is `text[left:right]`.

    Notes:
        - `left` is clamped to `0` and `right` is clamped to `len(text)`.
        - This function does not validate that `fo` offsets are within bounds; callers
          should ensure `fo.start_offset <= fo.end_offset` and both are valid indices.
    """
    left = max(0, fo.start_offset - window_left)
    right = min(len(text), fo.end_offset + window_right)
    seg = text[left:right]
    return left, right, seg


def _fo_occurrence_position(fo: FirstOccurrence, left: int) -> Span:
    """Return the first-occurrence span relative to a local window.

    Args:
        fo (FirstOccurrence): First occurrence with absolute offsets in the full text.
        left (int): Absolute start index of the local window in the full text.

    Returns:
        Span: `(start, end)` offsets relative to the local segment (`text[left:right]`).
    """
    return fo.start_offset - left, fo.end_offset - left


def _pick_better(best: Optional[ExtractedDefinition], cand: ExtractedDefinition) -> ExtractedDefinition:
    """Choose the better of two definition candidates.

    Args:
        best (ExtractedDefinition | None): Current best candidate.
        cand (ExtractedDefinition): New candidate to compare.

    Returns:
        ExtractedDefinition: The chosen candidate.
    """
    if best is None:
        return cand
    return (
        cand
        if cand.definition_confidence > best.definition_confidence
        else min((best, cand), key=lambda x: (-x.definition_confidence, (x.def_end - x.def_start)))
    )


def _anchored_confidence(*, base_conf: float, dist: float) -> float:
    """Compute anchored confidence with a small distance penalty.

    Applies a linear penalty of 0.0005 per character of distance, capped at
    200 characters, and caps the final confidence at 0.99.

    Args:
        base_conf (float): Base confidence for the matched pattern.
        dist (float): Character distance from the first occurrence.

    Returns:
        float: Confidence score in the range `(-inf, 0.99]`.
    """
    return min(base_conf - min(dist, 200) * 0.0005, 0.99)


def _distance_from_fo(*, a0_local: int, left: int, fo_start_offset: int) -> int:
    """Return absolute character distance from the first occurrence start.

    Args:
        a0_local (int): Acronym start offset within the local segment.
        left (int): Absolute start index of the local segment in the full text.
        fo_start_offset (int): Absolute start offset of the first occurrence.

    Returns:
        int: Absolute distance in characters.
    """
    return abs((a0_local + left) - fo_start_offset)


def extract_near_firsts(
    text: str,
    firsts: Mapping[str, FirstOccurrence],
    *,
    window_left: int,
    window_right: int,
    cfg: ExtractionConfig,
) -> dict[str, Optional[InTextPick]]:
    """Extract anchored in-text definitions near known first occurrences.

    For each first occurrence, builds a local text window, applies anchored
    acronym-definition patterns for the observed acronym surface, and selects the
    best cleaned definition candidate found near the first-occurrence span.

    Matching is anchored to the known first-occurrence offsets within the local
    window. When the detector preserved a trailing dot for a dotted initialism,
    the matcher tolerates a regex acronym span that excludes that final dot and
    normalises it back to the detector span before scoring.

    Args:
        text: Full source text.
        firsts: Mapping of normalised acronym key to first-occurrence metadata.
        window_left: Number of characters to include to the left of each first occurrence.
        window_right: Number of characters to include to the right of each first occurrence.
        cfg: ExtractionConfig() the extraction config for Acronyms.

    Returns:
        Mapping from each input key to the best nearby anchored definition pick, or
        ``None`` when no valid definition can be resolved for that first occurrence.

    Notes:
        - Candidate definitions are resolved from anchored regex matches, cleaned,
          scored by pattern kind and distance from the first occurrence, and then
          compared via ``_pick_better``.
        - The result preserves the input keys from ``firsts``.
    """

    picks: dict[str, Optional[InTextPick]] = {}

    for key, fo in firsts.items():
        acr_key = key or fo.acronym.upper()  # dict key / meaning key
        acr_surface = fo.acronym  # what actually appears in text window

        left, right, seg = _build_local_window(text, fo, window_left, window_right)

        fo_a0_local, fo_a1_local = _fo_occurrence_position(fo, left)

        best: Optional[ExtractedDefinition] = None

        for spec in compile_anchored_for_surface(acr_surface, cfg):
            pat = spec.pat
            kind = spec.kind
            strategy = spec.strategy

            for m in pat.finditer(seg):
                a0_local, a1_local = m.span("acr")

                # Require exact alignment with the known FO span.
                # BUT: in dotted_display="preserve", detector may extend FO to include a trailing '.' (U.S.A.)
                if a0_local != fo_a0_local or a1_local != fo_a1_local:
                    # Allow regex to capture without the trailing dot while FO includes it.
                    if (
                        a0_local == fo_a0_local
                        and a1_local + 1 == fo_a1_local
                        and 0 <= fo_a1_local - 1 < len(seg)
                        and seg[fo_a1_local - 1] == "."
                    ):
                        # Treat acronym end as the FO end (include the dot) so spans match detector occurrences.
                        a1_local = fo_a1_local
                    else:
                        continue
                span = resolve_def_span(strategy, seg=seg, m=m, acr_key=acr_key, a1_local=a1_local, cfg=cfg)
                if span is None:
                    continue
                d0_local, d1_local = span
                if d0_local >= d1_local:
                    continue

                # Original (pre-clean) definition slice from the segment
                orig = seg[d0_local:d1_local]

                clean = clean_definition(orig, acr_norm=acr_key, cfg=cfg, kind=kind)
                if clean is None:
                    continue

                dist = _distance_from_fo(a0_local=a0_local, left=left, fo_start_offset=fo.start_offset)
                base = base_for_kind(cfg, kind)
                conf = _anchored_confidence(base_conf=base, dist=dist)

                cand = ExtractedDefinition(
                    acronym=acr_key,
                    definition=clean,
                    source="first_occurrence_anchored",
                    definition_confidence=conf,
                    acr_start=a0_local + left,
                    acr_end=a1_local + left,
                    def_start=d0_local + left,
                    def_end=d1_local + left,
                    original_definition=orig,
                    kind=kind,
                )

                best = _pick_better(best, cand)

        picks[key] = (
            None
            if best is None
            else InTextPick(
                definition=best.definition,
                acr_span=(best.acr_start, best.acr_end),
                def_span=(best.def_start, best.def_end),
                definition_confidence=best.definition_confidence,
                original_definition=best.original_definition,
                kind=best.kind or "unknown",
            )
        )
    return picks
