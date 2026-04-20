import re
from typing import Optional

from document_resolution.nlp.common.constants_regex import TOKEN_RE
from document_resolution.nlp.common.shared import has_letter
from document_resolution.nlp.common.types import INLINE, INLINE_BEFORE, INLINE_KINDS
from document_resolution.nlp.extraction.acronyms.anchored.normalise import tighten_definition_span
from document_resolution.nlp.extraction.acronyms.config import ExtractionConfig
from document_resolution.nlp.extraction.acronyms.core.normalise import normalize_definition
from document_resolution.nlp.extraction.acronyms.matchers.tighten import tighten_label_by_acronym

_DET_PREFIX_RE = re.compile(r"^\s*(?:the|a|an)\b\s+", re.IGNORECASE)


def _strip_leading_determiner(s: str) -> str:
    """Strip a single leading English determiner from a string.
    Args:
        s (str): Input string that may begin with a determiner.

    Returns:
        str: String with one leading determiner removed, if present.
    """
    return _DET_PREFIX_RE.sub("", s, count=1)


def clean_definition(orig: str, *, acr_norm: str, cfg: ExtractionConfig, kind: str) -> Optional[str]:
    """Normalise and validate a candidate definition span.

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
    if not clean or not has_letter(clean) or len(clean) > cfg.max_phrase_chars:
        return None

    if cfg.require_two_words and kind in INLINE_KINDS and len(TOKEN_RE.findall(clean)) < 2:
        return None
    return clean
