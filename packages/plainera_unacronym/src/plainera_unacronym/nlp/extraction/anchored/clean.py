import re
from typing import Optional

from plainera_unacronym.nlp.common.constants_regex import TOKEN_RE
from plainera_unacronym.nlp.common.types import INLINE_KINDS, INLINE, INLINE_BEFORE
from plainera_unacronym.nlp.extraction import ExtractionConfig
from plainera_unacronym.nlp.extraction.anchored.normalise import tighten_definition_span
from plainera_unacronym.nlp.extraction.core.normalise import normalize_definition, has_letters
from plainera_unacronym.nlp.extraction.matchers.tighten import tighten_label_by_acronym

_DET_PREFIX_RE = re.compile(r"^\s*(?:the|a|an)\b\s+", re.IGNORECASE)

def _strip_leading_determiner(s: str) -> str:
    """Strip a single leading English determiner from a string.

    Removes exactly one leading determiner token ("the", "a", or "an") when it
    appears at the start of the string (ignoring leading whitespace). Matching
    is case-insensitive. If no leading determiner is present, returns the input
    unchanged.

    Args:
        s (str): Input string that may begin with a determiner.

    Returns:
        str: String with one leading determiner removed, if present.
    """
    return _DET_PREFIX_RE.sub("", s, count=1)



def clean_definition(orig: str, *, acr_norm: str, cfg: ExtractionConfig, kind: str) -> Optional[str]:
    """Normalise and validate a candidate definition span.

    Applies kind-specific pre-cleaning (e.g., tightening inline spans or stripping
    leading determiners), then runs tightening/normalisation and enforces guardrails:
    non-empty, contains letters, max length, and (optionally) minimum token count
    for inline kinds.

    Behaviour by `kind`:
        - INLINE: uses `tighten_definition_span(orig)` to prefer a definition-ish run.
        - INLINE_BEFORE: strips a leading determiner via `_strip_leading_determiner(orig)`.
        - Other kinds: uses `orig` as-is.

    Args:
        orig (str): Raw definition text extracted from the document.
        acr_norm (str): Normalised acronym key (typically uppercased).
        cfg (ExtractionConfig): Extraction configuration controlling limits and gates.
        kind (str): Definition kind (e.g., INLINE, INLINE_BEFORE, or other match kinds).

    Returns:
        Optional[str]: Normalised definition string if it passes validation, otherwise None.
    """
    if kind in INLINE_KINDS:
        raw = " ".join(orig.split())
        if len(raw) > cfg.max_phrase_chars:
            return None

    if kind == INLINE:
        base = tighten_definition_span(orig)
    elif kind == INLINE_BEFORE:
        base = _strip_leading_determiner(orig)
    else:
        base = orig

    clean = normalize_definition(tighten_label_by_acronym(base, acr_norm))
    if not clean or not has_letters(clean) or len(clean) > cfg.max_phrase_chars:
        return None

    if cfg.require_two_words and kind in INLINE_KINDS:
        if len(TOKEN_RE.findall(clean)) < 2:
            return None

    return clean
