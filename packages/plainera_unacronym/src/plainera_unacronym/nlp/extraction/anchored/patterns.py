import re

from plainera_unacronym.nlp.extraction import ExtractionConfig


def compile_anchored_exact(acr: str, cfg: ExtractionConfig):
    ACR = re.escape(acr)
    DEF = r"(?P<def>[^){}]{1,%d}?)" % cfg.max_phrase_chars

    # Allow "(ACR, ...)" or "(ACR; ...)" etc, but keep the ACR group tight.
    # Cap tail length to avoid runaway matches.
    TAIL = r"(?:\s*[,;:]\s*[^)]{0,%d})?" % min(120, cfg.max_phrase_chars)

    # Definition before (ACRONYM in parens) — supports "(ACR)" and "(ACR, tail)"
    fwd = re.compile(
        rf"\b{DEF}\s*\(\s*(?P<acr>{ACR}){TAIL}\s*\)",
        re.IGNORECASE | re.MULTILINE,
    )

    # Definition after (ACRONYM (definition)) — leave as-is
    rev = re.compile(
        rf"\b(?P<acr>{ACR})\s*\(\s*{DEF}\s*\)",
        re.IGNORECASE | re.MULTILINE,
    )

    #  inline: ACR ... cue ... DEF
    inlines_after = [
        re.compile(rf"\b(?P<acr>{ACR})\b\s*,?\s*{cue}\s+{DEF}", re.IGNORECASE | re.MULTILINE)
        for cue in cfg.inline_cues
    ]

    #  inline before: DEF ... cue ... ACR   (for "..., abbreviated as SLA,")
    inlines_before = [
        re.compile(
            rf"\b"
            rf"(?P<def>[^){{}}]{{1,{cfg.max_phrase_chars}}}?)"
            rf"(?=\s*,?\s*{cue}\s+{ACR}\b)"
            rf"\s*,?\s*{cue}\s+(?P<acr>{ACR})\b",
            re.IGNORECASE | re.MULTILINE,
        )
        for cue in cfg.inline_cues
    ]

    return (
        (fwd, cfg.conf_parenthetical, "def_before"),
        (rev, cfg.conf_parenthetical, "def_after"),
        *[(p, cfg.conf_inline, "inline") for p in inlines_after],
        *[(p, cfg.conf_inline, "inline_before") for p in inlines_before],
    )
