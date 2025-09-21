from __future__ import annotations
import re
from typing import Mapping, Optional, Pattern
from .config import ExtractionConfig
from .tighten import tighten_label_by_acronym
from ..common.shared import tighten_definition_span, tighten_label
from ..common.types import FirstOccurrence, ExtractedDefinition, InTextPick


def _acr_literal_pattern(acr_norm: str) -> str:
    # Match dotted or undotted variants (N.A.T.O. / NATO). Only letters get optional dots.
    parts = []
    for ch in acr_norm:
        if "A" <= ch <= "Z":
            parts.append(fr"{re.escape(ch)}\.?")
        else:
            parts.append(re.escape(ch))
    return "".join(parts)

def _compile_anchored(acr_norm: str, cfg: ExtractionConfig) -> list[tuple[Pattern[str], float, str]]:
    A = _acr_literal_pattern(acr_norm)
    P = rf"(?P<def>[^){{}}]{{1,{cfg.max_phrase_chars}}}?)"
    pats: list[tuple[str, float, str]] = [
        (rf"\b{P}\s*\(\s*(?P<acr>{A})\s*\)", cfg.conf_parenthetical, "paren-fwd"),
        (rf"\b(?P<acr>{A})\s*\(\s*{P}\s*\)",  cfg.conf_parenthetical, "paren-rev"),
    ]
    if cfg.enabled_inline:
        for cue in cfg.inline_cues:
            pats.append((rf"\b(?P<acr>{A})\b\s*,?\s*{cue}\s+{P}", cfg.conf_inline, "inline"))
    flags = re.IGNORECASE | re.MULTILINE
    return [(re.compile(p, flags), base, kind) for p, base, kind in pats]


def extract_near_firsts(
    text: str,
    firsts: Mapping[str, FirstOccurrence],  # key = normalized_key
    cfg: ExtractionConfig = ExtractionConfig(),
    window_left: int = 220,
    window_right: int = 280,
) -> dict[str, Optional[InTextPick]]:
    picks: dict[str, Optional[InTextPick]] = {}
    for key, fo in firsts.items():
        # Use the normalized key when anchoring; fallback to FO.acronym upper
        acr_norm = (key or fo.acronym.upper())
        L = max(0, fo.start_offset - window_left)
        R = min(len(text), fo.end_offset + window_right)
        seg = text[L:R]

        best: Optional[tuple[ExtractedDefinition, float, str]] = None
        for pat, base_conf, kind in _compile_anchored(acr_norm, cfg):
            for m in pat.finditer(seg):
                # local → global coords
                a0, a1 = m.span("acr"); d0, d1 = m.span("def")
                a0 += L; a1 += L; d0 += L; d1 += L
                # require that acronym overlap FO’s acronym span (guards cross-attachment)
                if a1 <= fo.start_offset or a0 >= fo.end_offset:
                    continue
                orig = m.group("def")
                clean = tighten_label(tighten_definition_span(orig))
                if not clean or len(clean) > cfg.max_phrase_chars:
                    continue
                # score: base + small distance penalty
                dist = abs(a0 - fo.start_offset)
                conf = min(base_conf - min(dist, 200) * 0.0005, 0.99)  # gently prefer nearer matches
                cand = ExtractedDefinition(
                    acronym=acr_norm,
                    definition=tighten_label_by_acronym(clean, acr_norm.upper()),
                    source="in_text",
                    confidence=conf,
                    acr_start=a0,
                    acr_end=a1,
                    def_start=d0,
                    def_end=d1,
                    original_definition=orig,
                )
                if (best is None) or (cand.confidence > best[0].confidence) \
                   or (cand.confidence == best[0].confidence and dist < abs(best[0].acr_start - fo.start_offset)):
                    best = (cand, conf, kind)

        picks[key] = None if best is None else InTextPick(
            definition=best[0].definition,
            acr_span=(best[0].acr_start, best[0].acr_end),
            def_span=(best[0].def_start, best[0].def_end),
            confidence=best[0].confidence,
            original_definition=best[0].original_definition,
        )
    return picks
