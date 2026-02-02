import re
from typing import Optional

from plainera_unacronym.nlp.common.constants_regex import TOKEN_RE
from plainera_unacronym.nlp.common.shared import normalize_definition, has_letters
from plainera_unacronym.nlp.extraction import ExtractionConfig
from plainera_unacronym.nlp.extraction.anchored.normalise import tighten_definition_span
from plainera_unacronym.nlp.extraction.matchers.tighten import tighten_label_by_acronym

_DET_PREFIX_RE = re.compile(r"^\s*(?:the|a|an)\b\s+", re.IGNORECASE)

def _strip_leading_determiner(s: str) -> str:
    return _DET_PREFIX_RE.sub("", s, count=1)

INLINE_KINDS = {"inline", "inline_before"}

def clean_definition(orig: str, *, acr_norm: str, cfg: ExtractionConfig, kind: str) -> Optional[str]:
    if kind in INLINE_KINDS:
        raw = " ".join(orig.split())
        if len(raw) > cfg.max_phrase_chars:
            return None

    if kind == "inline":
        base = tighten_definition_span(orig)
    elif kind == "inline_before":
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
