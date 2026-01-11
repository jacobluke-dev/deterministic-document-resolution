import re
import unicodedata

from plainera_unacronym.nlp.common.config import CANON_TABLE, TRAILING_PUNCT
from plainera_unacronym.nlp.common.constants import TITLECASE_RUN_RE, BOUNDARY_RE, _TITLE, _LINKERS_RE, _DASH


def canonicalize(s: str) -> str:
    # NFKC normalisation + map look-alikes (apostrophes, dashes)
    return unicodedata.normalize("NFKC", s).translate(CANON_TABLE)


def collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def strip_trailing_punct(s: str) -> str:
    return re.sub(TRAILING_PUNCT, "", s)


def tighten_definition_span(s: str) -> str:
    s = s.strip()
    if not s:
        return s

    # 1) Prefer the RIGHTMOST TitleCase run anywhere in the string
    last = None
    for m in TITLECASE_RUN_RE.finditer(s):
        last = m
    if last:
        return strip_trailing_punct(collapse_ws(last.group(1).strip()))

    # 2) Fallback: last clause, then try again for a rightmost run inside that clause
    parts = BOUNDARY_RE.split(s)
    tail = parts[-1].strip() if parts else s

    last_tail = None
    for m in TITLECASE_RUN_RE.finditer(tail):
        last_tail = m
    if last_tail:
        return strip_trailing_punct(collapse_ws(last_tail.group(1).strip()))

    # 2a) EXTRA safety: if tail starts with a TitleCase run, keep ONLY that run
    m_head = re.match(
        rf"^{_TITLE}(?:\s+(?:{_TITLE}|(?:{_LINKERS_RE}|{_DASH})\s+{_TITLE}))*",
        tail,
        flags=re.UNICODE,
    )
    if m_head:
        return strip_trailing_punct(collapse_ws(m_head.group(0)))

    # 3) Final fallback: just clean the tail
    return strip_trailing_punct(collapse_ws(tail))



def normalize_definition(s: str) -> str:
    """
    UX/display normalisation for definitions:
      - NFKC + fold dash/apostrophes
      - collapse whitespace
      - strip trailing punctuation
    """
    return strip_trailing_punct(collapse_ws(canonicalize(s)))


def clean_and_validate(raw_slice, acr, cfg, *, kind) -> str | None:
    """
        enforces require_two_words

        enforces length gates (ideally on both raw and cleaned where appropriate)
    """
    pass
