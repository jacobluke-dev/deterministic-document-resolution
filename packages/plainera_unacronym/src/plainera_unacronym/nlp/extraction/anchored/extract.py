import re
from typing import Mapping, Optional

from plainera_unacronym.nlp import FirstOccurrence
from plainera_unacronym.nlp.common.shared import normalize_definition
from plainera_unacronym.nlp.common.types import InTextPick, ExtractedDefinition
from plainera_unacronym.nlp.extraction import ExtractionConfig
from plainera_unacronym.nlp.extraction.anchored.normalise import tighten_definition_span
from plainera_unacronym.nlp.extraction.anchored.patterns import compile_anchored_exact
from plainera_unacronym.nlp.extraction.matchers.helper_patterns import (find_parenthetical_longform_after_acr,
                                                                        find_parenthetical_longform_before_acr,
                                                                        find_inline_longform_after_acr,
                                                                        is_acronym_parenthetical_with_tail)
from plainera_unacronym.nlp.extraction.matchers.tighten import tighten_label_by_acronym

Span = tuple[int, int]
OptSpan = Optional[Span]

_TOKEN_RE = re.compile(r"[A-Za-z0-9][\w’'\-]*")


def _clean_definition(orig: str, *, acr_norm: str, cfg: ExtractionConfig, kind: str) -> Optional[str]:
    # Inline-only raw length gate (before tightening)
    if kind == "inline":
        raw = " ".join(orig.split())  # collapse whitespace
        if len(raw) > cfg.max_phrase_chars:
            return None

    # Only inline needs span-tightening; parentheticals are already tight.
    base = tighten_definition_span(orig) if kind == "inline" else orig

    clean = tighten_label_by_acronym(base, acr_norm)
    clean = normalize_definition(clean)

    if not clean or len(clean) > cfg.max_phrase_chars:
        return None

    if cfg.require_two_words and kind == "inline":
        if len(_TOKEN_RE.findall(clean)) < 2:
            return None

    return clean


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


def _fo_occurrence_position(fo: FirstOccurrence, left: int) -> tuple[int, int]:
    # FO position in the local segment
    return fo.start_offset - left, fo.end_offset - left


def _span_of_pre_definition(seg: str, paren_start: int) -> tuple[int, int] | None:
    """
    Return (start, end) span for definition immediately before '(' within seg.
    End is right-trimmed to avoid capturing the space before '('.
    """
    end = len(seg[:paren_start].rstrip())
    if end <= 0:
        return None
    return 0, end


def _calc_def_span(kind: str, *, acr_norm: str, seg: str, acr_end_local: int = None, m: re.Match[str] = None,
                   cfg: ExtractionConfig,
                   ) -> OptSpan:
    if kind == "def_after":
        snippet = seg[acr_end_local:]
        mm = find_parenthetical_longform_after_acr(snippet, cfg, acr=acr_norm, require_initials_match=True)
        if not mm:
            return None
        loc = mm[0]
        return acr_end_local + loc.def_start, acr_end_local + loc.def_end

    if kind == "def_before":
        # NEW: detect "(ACR, ...)" and treat definition as BEFORE '('
        if is_acronym_parenthetical_with_tail(seg[m.start():], acr_norm):
            return _span_of_pre_definition(seg, m.start())

        # Existing behaviour: "Long Form ... (ACR)" anchored
        snippet = seg[: m.end()]
        mm = find_parenthetical_longform_before_acr(snippet, acr_norm, cfg)
        if not mm:
            return None
        loc = mm[0]
        return loc.def_start, loc.def_end

    # inline
    snippet = seg[acr_end_local:]
    mm = find_inline_longform_after_acr(
        snippet, cfg, acr=acr_norm, max_chars=cfg.max_phrase_chars * 2, require_initials_match=True
    )
    if not mm:
        return None
    loc = mm[0]
    return acr_end_local + loc.def_start, acr_end_local + loc.def_end


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
        acr_norm = key or fo.acronym.upper()

        left, right, seg = _build_local_window(text, fo, window_left, window_right)

        fo_a0_local, fo_a1_local = _fo_occurrence_position(fo, left)

        best: Optional[ExtractedDefinition] = None

        for pat, base_conf, kind in compile_anchored_exact(acr_norm, cfg):

            for m in pat.finditer(seg):
                a0_local, a1_local = m.span("acr")
                # Require exact alignment with the known FO span
                if a0_local != fo_a0_local or a1_local != fo_a1_local:
                    continue


                if kind == "def_after":
                    span = _calc_def_span('def_after', acr_norm=acr_norm, seg=seg, acr_end_local=a1_local, cfg=cfg)
                    if span is None:
                        continue
                    d0_local, d1_local = span
                    if d0_local >= d1_local:
                        continue

                elif kind == "def_before":
                    span = _calc_def_span('def_before', acr_norm=acr_norm, seg=seg, m=m, cfg=cfg)
                    if span is None:
                        continue
                    d0_local, d1_local = span
                    if d0_local >= d1_local:
                        continue

                elif kind == "paren_before_acr":
                    d0_local, d1_local = m.span("def")
                    if d0_local >= d1_local:
                        continue

                elif kind == "inline_before":
                    d0_local, d1_local = m.span("def")
                    if d0_local >= d1_local:
                        continue

                else:  # "inline" → look-ahead initials alignment (no parentheses)

                    span = _calc_def_span('inline', acr_norm=acr_norm, seg=seg, acr_end_local=a1_local, cfg=cfg)
                    if span is None:
                        continue
                    d0_local, d1_local = span
                    if d0_local >= d1_local:
                        continue

                # Original (pre-clean) definition slice from the segment
                orig = seg[d0_local:d1_local]

                if kind in {"inline", "inline_before"}:
                    raw = " ".join(orig.split())
                    if len(raw) > cfg.max_phrase_chars:
                        continue

                clean = _clean_definition(orig, acr_norm=acr_norm, cfg=cfg, kind=kind)

                if cfg.require_two_words and kind in {"inline", "inline_before"}:
                    if len(_TOKEN_RE.findall(clean)) < 2:
                        continue
                if clean is None:
                    continue

                # Confidence — distance is 0 at FO, but keep the formula
                dist = _distance_from_fo(a0_local=a0_local, left=left, fo_start_offset=fo.start_offset)
                conf = _anchored_confidence(base_conf=base_conf, dist=dist)

                cand = ExtractedDefinition(
                    acronym=acr_norm,
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
