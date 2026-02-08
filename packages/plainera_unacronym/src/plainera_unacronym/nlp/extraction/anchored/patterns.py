import re
from dataclasses import dataclass
from typing import Any

from plainera_unacronym.nlp.common.constants_regex import QUOTE as QUOTE_RE
from plainera_unacronym.nlp.common.types import Definition_strategy
from plainera_unacronym.nlp.extraction import ExtractionConfig


@dataclass(frozen=True, slots=True)
class PatternSpec:
    pat: re.Pattern[str]
    base_conf: float
    strategy: Definition_strategy
    kind: str


def compile_anchored_exact(
    acr: str, cfg: ExtractionConfig
) -> tuple[PatternSpec, PatternSpec, PatternSpec, PatternSpec, PatternSpec, PatternSpec, Any, Any]:
    """Compile anchored extraction patterns for a specific acronym.

    Builds a set of compiled regex patterns that detect common long-form/acronym
    structures around an exact acronym surface, including:

      - Forward wrappers:  Long Form (ACR) / Long Form [ACR]
      - Reverse wrappers:  ACR (Long Form) / ACR [Long Form]
      - Wrapper-before-acr: (Long Form) ACR / [Long Form] ACR
      - Inline cues: ACR ... <cue> ... Long Form  and  Long Form ... <cue> ... ACR

    Each pattern exposes named capture groups:
      - ``acr``: the acronym only (excluding optional trailing dot, quotes, tails,
        and optional possessive markers like ``PDF's`` / ``PDF’s``).
      - ``def``: a candidate definition region (may be intentionally minimal for
        some inline patterns; use ``resolve_def_span(spec.strategy, ...)`` to
        compute the final span to slice).

    Args:
        acr: Exact acronym surface to compile patterns for (e.g., ``"PPE"``).
        cfg: Extraction configuration controlling phrase limits, inline cue
            phrases, and base confidence values.

    Returns:
        Tuple of PatternSpec instances. Order reflects intended matching priority
        (wrapper forms first, then inline forms). Each PatternSpec includes a
        compiled regex, base confidence, strategy identifier (for
        ``resolve_def_span``), and a kind label used for downstream cleaning.
    """
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
        re.compile(rf"\b(?P<acr>{ACR})\b\s*,?\s*{cue}\s+{DEF}", re.IGNORECASE | re.MULTILINE) for cue in cfg.inline_cues
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
        PatternSpec(fwd_paren, cfg.conf_parenthetical, "helper_def_before", "def_before"),
        PatternSpec(fwd_brack, cfg.conf_parenthetical, "direct_def", "def_before_direct"),
        PatternSpec(rev_paren, cfg.conf_parenthetical, "helper_def_after", "def_after"),
        PatternSpec(rev_brack, cfg.conf_parenthetical, "direct_def", "def_after_direct"),
        PatternSpec(before_acr_paren, cfg.conf_parenthetical, "direct_def", "before_acr_paren"),
        PatternSpec(before_acr_brack, cfg.conf_parenthetical, "direct_def", "paren_before_acr"),
        *[PatternSpec(p, cfg.conf_inline, "helper_inline_after", "inline") for p in inlines_after],
        *[PatternSpec(p, cfg.conf_inline, "direct_def", "inline_before") for p in inlines_before],
    )
