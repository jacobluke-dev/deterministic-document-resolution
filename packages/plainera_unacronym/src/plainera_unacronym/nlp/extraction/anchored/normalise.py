import re

from plainera_unacronym.nlp.common.constants_regex import (TITLECASE_RUN_RE,
                                                           BOUNDARY_RE,
                                                           _TITLELIKE,
                                                           _LINKERS_RE,
                                                           _DASH)
from plainera_unacronym.nlp.common.shared import strip_trailing_punct, collapse_ws


def tighten_definition_span(s: str) -> str:
    s = s.strip()
    print("TITLECASE_RUN_RE:", s)
    if not s:
        return s

    # 1) Prefer the RIGHTMOST TitleCase run anywhere in the string
    last = None
    for m in TITLECASE_RUN_RE.finditer(s):
        last = m
    if last:
        print("TITLECASE_RUN_RE:", last)
        return strip_trailing_punct(collapse_ws(last.group(1).strip()))

    # 2) Fallback: last clause, then try again for a rightmost run inside that clause
    parts = BOUNDARY_RE.split(s)
    tail = parts[-1].strip() if parts else s

    last_tail = None
    for m in TITLECASE_RUN_RE.finditer(tail):
        last_tail = m
    if last_tail:
        print("last_tail:", last_tail)
        return strip_trailing_punct(collapse_ws(last_tail.group(1).strip()))

    # 2a) EXTRA safety: if tail starts with a TitleCase run, keep ONLY that run
    m_head = re.match(
        rf"^{_TITLELIKE}(?:\s+(?:{_TITLELIKE}|{_LINKERS_RE}\s+{_TITLELIKE}|{_DASH}\s+{_TITLELIKE}))*",
        tail,
        flags=re.UNICODE,
    )
    if m_head:
        # print("m_head:", m_head)
        return strip_trailing_punct(collapse_ws(m_head.group(0)))
    print("final return ", tail)
    # 3) Final fallback: just clean the tail
    return strip_trailing_punct(collapse_ws(tail))
