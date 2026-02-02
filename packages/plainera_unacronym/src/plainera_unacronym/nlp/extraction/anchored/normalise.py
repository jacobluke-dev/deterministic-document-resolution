import re

from plainera_unacronym.nlp.common.constants_regex import (TITLECASE_RUN_RE,
                                                           BOUNDARY_RE,
                                                           _TITLELIKE,
                                                           _LINKERS_RE,
                                                           _DASH, _TITLE)
from plainera_unacronym.nlp.common.shared import strip_trailing_punct_str, collapse_ws


_TITLE_TOKEN_RE = re.compile(rf"(?:^|\s){_TITLE}(?:$|\s)", flags=re.UNICODE)


def tighten_definition_span(s: str) -> str:
    """Tighten a noisy definition string down to its most plausible “definition-ish” span.

        The function prefers the rightmost TitleCase/ALLCAPS run found by TITLECASE_RUN_RE,
        but avoids selecting a pure lowercase-hyphen tail (e.g., "sign-on") as the result.
        If no acceptable run is found, it falls back to analysing the last clause and then
        finally returns a whitespace-collapsed, trailing-punctuation-trimmed tail.

        Args:
            s: Raw definition string (may include punctuation, multiple clauses, etc.).

        Returns:
            A tightened string with collapsed whitespace and trailing punctuation removed.
            Returns "" if input is blank/whitespace.
    """
    s = s.strip()
    if not s:
        return s

    # 1) Prefer the RIGHTMOST TitleCase run anywhere in the string
    last = None
    for m in TITLECASE_RUN_RE.finditer(s):
        last = m
    if last:
        candidate = last.group(1).strip()

        # NEW: don't let a pure lowercase-hyphen tail (e.g., "sign-on") win
        if _TITLE_TOKEN_RE.search(candidate):
            return strip_trailing_punct_str(collapse_ws(candidate))
        # else: ignore this match and fall through

    # 2) Fallback: last clause, then try again for a rightmost run inside that clause
    parts = BOUNDARY_RE.split(s)
    tail = parts[-1].strip() if parts else s

    last_tail = None
    for m in TITLECASE_RUN_RE.finditer(tail):
        last_tail = m
    if last_tail:
        candidate = last_tail.group(1).strip()
        if _TITLE_TOKEN_RE.search(candidate):
            return strip_trailing_punct_str(collapse_ws(candidate))

    # 2a) EXTRA safety: if tail starts with a TitleCase run, keep ONLY that run
    m_head = re.match(
        rf"^{_TITLELIKE}(?:\s+(?:{_TITLELIKE}|{_LINKERS_RE}\s+{_TITLELIKE}|{_DASH}\s+{_TITLELIKE}))*",
        tail,
        flags=re.UNICODE,
    )
    if m_head:
        return strip_trailing_punct_str(collapse_ws(m_head.group(0)))
    # 3) Final fallback: just clean the tail
    return strip_trailing_punct_str(collapse_ws(tail))
