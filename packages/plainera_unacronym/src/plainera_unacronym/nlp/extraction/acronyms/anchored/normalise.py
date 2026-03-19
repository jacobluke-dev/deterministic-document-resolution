import re

from plainera_unacronym.nlp.common.constants_regex import (
    _DASH,
    _LINKERS_RE,
    _TITLE,
    _TITLELIKE,
    BOUNDARY_RE,
    TOKEN_RE,
)
from plainera_unacronym.nlp.common.shared import collapse_ws, strip_trailing_punct_str

_TITLE_TOKEN_RE = re.compile(rf"(?:^|[\s,;:()]){_TITLE}(?=$|[\s,;:().])", flags=re.UNICODE)

_TITLECASE_RUN_ANY_RE = re.compile(
    rf"(?:^|[\s,])"
    rf"("
    rf"{_TITLELIKE}"
    rf"(?:\s+(?:"
    rf"{_TITLELIKE}"
    rf"|{_LINKERS_RE}\s+{_TITLELIKE}"
    rf"|{_DASH}\s+{_TITLELIKE}"
    rf"))*"
    rf")",
    flags=re.UNICODE,
)


def _pick_best_run(text: str) -> str | None:
    """
    Select the “best” title-case run from `text`.

    This scans `text` for title-cased spans matched by `_TITLECASE_RUN_ANY_RE`,
    filters out empty/degenerate candidates (e.g. spans that contain no valid
    title tokens per `_TITLE_TOKEN_RE`), and then chooses a winner using a
    deterministic ranking:

      1) Prefer the candidate with the greatest number of tokens (as counted by `TOKEN_RE`).
      2) If tied on token count, prefer the rightmost candidate (largest match end offset).

    This heuristic is designed to pick the most information-dense and most
    context-relevant title-case phrase when multiple runs are present.

    Args:
        text: Source text to scan for title-case runs.

    Returns:
        The selected title-case span (stripped), or None if no suitable
        candidate is found.
    """
    best: str | None = None
    best_key: tuple[int, int] | None = None  # (token_count, end_pos)

    for m in _TITLECASE_RUN_ANY_RE.finditer(text):
        cand = m.group(1).strip()
        if not cand:
            continue
        # prevent "sign-on" only (and similar) from winning
        if not _TITLE_TOKEN_RE.search(cand):
            continue

        tok_n = len(TOKEN_RE.findall(cand))
        key = (tok_n, m.end())  # prefer longer; tie-break by rightmost
        if best is None or best_key is None or key > best_key:
            best, best_key = cand, key

    return best


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
        return ""

    # 1) Prefer best run anywhere in the full string
    cand = _pick_best_run(s)
    if cand:
        return strip_trailing_punct_str(collapse_ws(cand))

    # 2) Fallback: last clause and try again
    parts = BOUNDARY_RE.split(s)
    tail = parts[-1].strip() if parts else s

    cand = _pick_best_run(tail)
    if cand:
        return strip_trailing_punct_str(collapse_ws(cand))

    # 3) Final fallback: collapsed tail
    return strip_trailing_punct_str(collapse_ws(tail))
