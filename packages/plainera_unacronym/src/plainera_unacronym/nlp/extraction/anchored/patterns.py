import re
from plainera_unacronym.nlp.extraction import ExtractionConfig
from plainera_unacronym.nlp.common.constants_regex import QUOTE as QUOTE_RE


def compile_anchored_exact(acr: str, cfg: ExtractionConfig):
    ACR = re.escape(acr)

    DOT = r"(?:\.)?"

    # Safe for (...) and [...]
    DEF = rf"(?P<def>[^\)\]\{{\}}]{{1,{cfg.max_phrase_chars}}}?)"

    # Allow tails after acronym inside wrapper: (PPE, ...), (PPE - ...), [PPE: ...]
    TAIL = rf"(?:\s*[,;:—–-]\s*[^\)\]]{{0,{min(120, cfg.max_phrase_chars)}}})?"

    # Allow possessive surfaces like: PDF's (Long Form) or PDF’s (Long Form)
    # IMPORTANT: keep (?P<acr>...) as ONLY the acronym, so FO spans still align.
    JOIN_POSSESSIVE = r"(?:\s*(?:['’]s)\b)?\s*"

    # Long Form (ACR...)  / Long Form [ACR...]
    fwd_paren = re.compile(
        rf"\b{DEF}\s*\(\s*{QUOTE_RE}(?P<acr>{ACR}){DOT}{QUOTE_RE}{TAIL}\s*\)",
        re.IGNORECASE | re.MULTILINE,
    )
    fwd_brack = re.compile(
        rf"\b{DEF}\s*\[\s*{QUOTE_RE}(?P<acr>{ACR}){DOT}{QUOTE_RE}{TAIL}\s*\]",
        re.IGNORECASE | re.MULTILINE,
    )

    # ACR (Long Form) / ACR [Long Form]
    # NOTE: allow optional possessive between acronym and wrapper.
    rev_paren = re.compile(
        rf"\b{QUOTE_RE}(?P<acr>{ACR}){DOT}{QUOTE_RE}\b{JOIN_POSSESSIVE}\(\s*{DEF}\s*\)",
        re.IGNORECASE | re.MULTILINE,
    )
    rev_brack = re.compile(
        rf"\b{QUOTE_RE}(?P<acr>{ACR}){DOT}{QUOTE_RE}\b{JOIN_POSSESSIVE}\[\s*{DEF}\s*\]",
        re.IGNORECASE | re.MULTILINE,
    )

    # (Long Form) ACR / [Long Form] ACR
    before_acr_paren = re.compile(
        rf"\(\s*{DEF}\s*\)\s+{QUOTE_RE}(?P<acr>{ACR}){DOT}{QUOTE_RE}\b",
        re.IGNORECASE | re.MULTILINE,
    )
    before_acr_brack = re.compile(
        rf"\[\s*{DEF}\s*\]\s+{QUOTE_RE}(?P<acr>{ACR}){DOT}{QUOTE_RE}\b",
        re.IGNORECASE | re.MULTILINE,
    )

    inlines_after = [
        re.compile(rf"\b(?P<acr>{ACR})\b\s*,?\s*{cue}\s+{DEF}", re.IGNORECASE | re.MULTILINE)
        for cue in cfg.inline_cues
    ]

    # inline before: DEF ... cue ... ACR
    inlines_before = [
        re.compile(
            rf"\b"
            rf"(?P<def>[^\)\]\{{\}}]{{1,{cfg.max_phrase_chars}}}?)"
            rf"(?=\s*,?\s*{cue}\s+{ACR}\b)"
            rf"\s*,?\s*{cue}\s+(?P<acr>{ACR})\b",
            re.IGNORECASE | re.MULTILINE,
        )
        for cue in cfg.inline_cues
    ]

    return (
        (fwd_paren, cfg.conf_parenthetical, "def_before"),
        (fwd_brack, cfg.conf_parenthetical, "def_before_direct"),
        (rev_paren, cfg.conf_parenthetical, "def_after"),
        (rev_brack, cfg.conf_parenthetical, "def_after_direct"),
        (before_acr_paren, cfg.conf_parenthetical, "before_acr_paren"),
        (before_acr_brack, cfg.conf_parenthetical, "paren_before_acr"),
        *[(p, cfg.conf_inline, "inline") for p in inlines_after],
        *[(p, cfg.conf_inline, "inline_before") for p in inlines_before],
    )
