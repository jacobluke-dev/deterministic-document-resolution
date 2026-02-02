from typing import Mapping, Optional

from plainera_unacronym.nlp import FirstOccurrence
from plainera_unacronym.nlp.common.types import InTextPick, ExtractedDefinition, Span
from plainera_unacronym.nlp.extraction import ExtractionConfig
from plainera_unacronym.nlp.extraction.anchored.clean import clean_definition
from plainera_unacronym.nlp.extraction.anchored.patterns import compile_anchored_exact
from plainera_unacronym.nlp.extraction.anchored.spans import resolve_def_span



def _build_local_window(
    text: str,
    fo: FirstOccurrence,
    window_left: int,
    window_right: int,
) -> tuple[int, int, str]:
    """Build a bounded local text window around a first occurrence.

    Computes a `[left:right]` slice around `fo` using the provided window sizes,
    clamped to the document bounds. Returns both absolute indices and the sliced
    segment for downstream matching.

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

        Selection rules:
            1) If `best` is None, return `cand`.
            2) Prefer higher `confidence`.
            3) If confidence ties, prefer the shorter definition span
               (`def_end - def_start`).

        Args:
            best (ExtractedDefinition | None): Current best candidate.
            cand (ExtractedDefinition): New candidate to compare.

        Returns:
            ExtractedDefinition: The chosen candidate.
        """
    if best is None:
        return cand
    return cand if cand.confidence > best.confidence else min((best, cand),
                                                              key=lambda x: (-x.confidence, (x.def_end - x.def_start)))


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

        Converts a local segment offset back to an absolute offset by adding `left`,
        then returns the absolute difference from `fo_start_offset`.

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
    cfg: ExtractionConfig = ExtractionConfig(),
) -> dict[str, Optional[InTextPick]]:
    picks: dict[str, Optional[InTextPick]] = {}

    for key, fo in firsts.items():
        acr_key = key or fo.acronym.upper()  # dict key / sense key
        acr_surface = fo.acronym  # what actually appears in text window

        left, right, seg = _build_local_window(text, fo, window_left, window_right)

        fo_a0_local, fo_a1_local = _fo_occurrence_position(fo, left)

        best: Optional[ExtractedDefinition] = None

        for spec in compile_anchored_exact(acr_surface, cfg):
            pat = spec.pat
            base_conf = spec.base_conf
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

                # Confidence — distance is 0 at FO, but keep the formula
                dist = _distance_from_fo(a0_local=a0_local, left=left, fo_start_offset=fo.start_offset)
                conf = _anchored_confidence(base_conf=base_conf, dist=dist)

                cand = ExtractedDefinition(
                    acronym=acr_key,
                    definition=clean,
                    source="in_text",
                    confidence=conf,
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
                confidence=best.confidence,
                original_definition=best.original_definition,
                kind=best.kind or "unknown"
            )
        )
    return picks
