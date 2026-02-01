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
    # Build a local window around the first occurrence
    left = max(0, fo.start_offset - window_left)
    right = min(len(text), fo.end_offset + window_right)
    seg = text[left:right]
    return left, right, seg


def _fo_occurrence_position(fo: FirstOccurrence, left: int) -> Span:
    # FO position in the local segment
    return fo.start_offset - left, fo.end_offset - left


def _pick_better(best: Optional[ExtractedDefinition], cand: ExtractedDefinition) -> ExtractedDefinition:
    if best is None:
        return cand
    return cand if cand.confidence > best.confidence else min((best, cand),
                                                              key=lambda x: (-x.confidence, (x.def_end - x.def_start)))


def _anchored_confidence(*, base_conf: float, dist: float) -> float:
    return min(base_conf - min(dist, 200) * 0.0005, 0.99)


def _distance_from_fo(*, a0_local: int, left: int, fo_start_offset: int) -> int:
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
