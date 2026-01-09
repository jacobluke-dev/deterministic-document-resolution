import re

from plainera_unacronym.nlp.extraction import ExtractionConfig


def compile_anchored_exact(acr: str, cfg: ExtractionConfig):
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
