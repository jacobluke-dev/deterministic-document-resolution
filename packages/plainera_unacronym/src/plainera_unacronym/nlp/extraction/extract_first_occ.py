import re
from typing import Mapping, Optional

from .helper_patterns import find_parenthetical_longform_after_acr, find_parenthetical_longform_before_acr, \
    find_inline_longform_after_acr
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

        # Build a local window around the first occurrence
        left = max(0, fo.start_offset - window_left)
        right = min(len(text), fo.end_offset + window_right)
        seg = text[left:right]

        # FO position in the local segment
        fo_a0_local = fo.start_offset - left
        fo_a1_local = fo.end_offset - left

        best: Optional[ExtractedDefinition] = None

        for pat, base_conf, kind in _compile_anchored_exact(acr_norm, cfg):
            for m in pat.finditer(seg):
                a0_local, a1_local = m.span("acr")
                # Require exact alignment with the known FO span
                if a0_local != fo_a0_local or a1_local != fo_a1_local:
                    continue

                # We will compute (d0_local, d1_local) and take `orig = seg[d0_local:d1_local]`
                d0_local = d1_local = None

                if kind == "def_after":
                    # ACR … (DEF)  → parse parenthetical right AFTER the acronym
                    snippet = seg[a1_local:]  # helper expects snippet starting at/after ACR
                    mm = find_parenthetical_longform_after_acr(
                        snippet, cfg, acr=acr_norm, require_initials_match=True
                    )
                    if not mm:
                        continue
                    loc = mm[0]
                    d0_local = a1_local + loc.def_start
                    d1_local = a1_local + loc.def_end

                elif kind == "def_before":
                    # DEF … (ACR)  → parse text that ENDS with "(ACR)".
                    # Use the entire regex match so the closing paren is included.
                    snippet = seg[: m.end()]  # m.end() includes the trailing “)”
                    mm = find_parenthetical_longform_before_acr(snippet, acr_norm, cfg)
                    if not mm:
                        continue
                    loc = mm[0]
                    d0_local = loc.def_start
                    d1_local = loc.def_end


                else:  # "inline" → look-ahead initials alignment (no parentheses)

                    snippet = seg[a1_local:]  # start right after the acronym

                    mm = find_inline_longform_after_acr(

                        snippet, cfg, acr=acr_norm, max_chars=cfg.max_phrase_chars * 2, require_initials_match=True

                    )

                    if not mm:
                        continue

                    loc = mm[0]

                    d0_local = a1_local + loc.def_start

                    d1_local = a1_local + loc.def_end

                    orig = seg[d0_local:d1_local]

                    clean = loc.definition

                # Guard: spans must be valid
                if d0_local is None or d1_local is None or d0_local >= d1_local:
                    continue

                # Original (pre-clean) definition slice from the segment
                orig = seg[d0_local:d1_local]

                # Clean and normalize
                clean = tighten_label_by_acronym(
                    tighten_definition_span(orig),
                    acr_norm,
                )
                clean = normalize_definition(clean)

                # Optionally require at least two tokens (prevents "a"/"of", etc. for inline)
                if cfg.require_two_words:
                    if len(re.findall(r"[A-Za-z0-9][\w’'\-]*", clean)) < 2:
                        # Only enforce this for inline; parenthetical matches are usually fine.
                        if kind == "inline":
                            continue

                # Length gate
                if not clean or len(clean) > cfg.max_phrase_chars:
                    continue

                # Confidence — distance is 0 at FO, but keep the formula
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
