import re
from typing import Mapping, Optional

from ..common.shared import normalize_definition, tighten_definition_span
from ..common.types import ExtractedDefinition, FirstOccurrence, InTextPick
from .config import ExtractionConfig
from .tighten import tighten_label_by_acronym


def _compile_anchored_exact(acr: str, cfg: ExtractionConfig):
    ACR = re.escape(acr)
    DEF = r"(?P<def>[^){}]{1,%d}?)" % cfg.max_phrase_chars

    # Definition before (ACRONYM in parens)
    fwd = re.compile(rf"\b{DEF}\s*\(\s*(?P<acr>{ACR})\s*\)", re.IGNORECASE | re.MULTILINE)
    # Definition after (ACRONYM (definition))
    rev = re.compile(rf"\b(?P<acr>{ACR})\s*\(\s*{DEF}\s*\)", re.IGNORECASE | re.MULTILINE)

    inlines = [
        re.compile(rf"\b(?P<acr>{ACR})\b\s*,?\s*{cue}\s+{DEF}", re.IGNORECASE | re.MULTILINE) for cue in cfg.inline_cues
    ]
    return (
        (fwd, cfg.conf_parenthetical, "def_before"),
        (rev, cfg.conf_parenthetical, "def_after"),
        *[(p, cfg.conf_inline, "inline") for p in inlines],
    )


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

        left = max(0, fo.start_offset - window_left)
        right = min(len(text), fo.end_offset + window_right)
        seg = text[left:right]

        # FO position in the local segment
        fo_a0_local = fo.start_offset - left
        fo_a1_local = fo.end_offset - left

        best: Optional[ExtractedDefinition] = None
        for pat, base_conf, _kind in _compile_anchored_exact(acr_norm, cfg):
            for m in pat.finditer(seg):
                a0_local, a1_local = m.span("acr")
                # Require exact alignment with the known FO span
                if a0_local != fo_a0_local or a1_local != fo_a1_local:
                    continue

                d0_local, d1_local = m.span("def")
                orig = m.group("def")

                clean = tighten_label_by_acronym(tighten_definition_span(orig), acr_norm)

                clean = normalize_definition(clean)
                if not clean or len(clean) > cfg.max_phrase_chars:
                    continue

                # distance is effectively 0 when it’s the FO itself, but keep formula for consistency
                dist = abs((a0_local + left) - fo.start_offset)
                conf = min(base_conf - min(dist, 200) * 0.0005, 0.99)

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
                )
                if (best is None) or (cand.confidence > best.confidence):
                    best = cand

        picks[key] = (
            None
            if best is None
            else InTextPick(
                definition=best.definition,
                acr_span=(best.acr_start, best.acr_end),
                def_span=(best.def_start, best.def_end),
                confidence=best.confidence,
                original_definition=best.original_definition,
            )
        )
    return picks
