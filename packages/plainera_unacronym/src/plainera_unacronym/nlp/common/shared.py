import re
import unicodedata

from plainera_unacronym.nlp.common.config import CANON_TABLE, TRAILING_PUNCT
from plainera_unacronym.nlp.common.constants import ARTICLE, BOUNDARY_RE, LEADING_CONNECTORS, TITLECASE_RUN_RE, _TITLE, \
    _LINKERS_RE, _DASH


def has_paren_definition(text: str, end: int, max_chars: int = 80) -> bool:
    """Return whether a parenthetical definition follows immediately after a token.

    Starting at ``end`` (the character index directly after a candidate token),
    this checks for optional whitespace, then a ``'('`` and scans up to
    ``max_chars`` characters **inside** the parentheses looking for a closing
    ``')'``. It counts alphabetic characters within the parentheses and returns
    ``True`` only if a closing parenthesis is found within the limit and at
    least 5 letters are present (letters are determined via ``str.isalpha()``,
    so Unicode letters like Greek count).

    Args:
      text: The full source text being analyzed.
      end: The index in ``text`` immediately after the token to test.
      max_chars: Maximum number of characters to scan **inside** the parentheses
        before giving up. The closing ``')'`` must appear within this window.

    Returns:
      True if a parenthetical definition is detected within the limit and
      contains at least 5 letters; otherwise False.

    Examples:
      >>> s = "GPU (Graphics Processing Unit) is common."
      >>> has_paren_definition(s, s.index("GPU") + 3)
      True
      >>> s = "CPU (org) used here."
      >>> has_paren_definition(s, s.index("CPU") + 3)
      False
      >>> s = "ATP (αβγδε) binding"
      >>> has_paren_definition(s, s.index("ATP") + 3)
      True
    """
    i, n = end, len(text)
    # skip whitespace
    while i < n and text[i].isspace():
        i += 1
    # must have '(' next
    if not (i < n and text[i] == "("):
        return False

    j, alpha = i + 1, 0
    limit = i + 1 + max_chars  # scan at most max_chars chars *inside* the parens

    while j < n and j <= limit and text[j] != ")":
        if text[j].isascii() and text[j].isalpha():
            alpha += 1
        j += 1

    # valid only if we hit a closing ')' within the limit
    return (j < n and text[j] == ")") and (alpha >= 5)


def canonicalize(s: str) -> str:
    # NFKC normalisation + map look-alikes (apostrophes, dashes)
    return unicodedata.normalize("NFKC", s).translate(CANON_TABLE)


def collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def strip_trailing_punct(s: str) -> str:
    return re.sub(TRAILING_PUNCT, "", s)


def _swallow_spaces_around_allowed(s: str, allow_chars: str) -> str:
    # R & D -> R&D ; keep everything else unchanged
    if not allow_chars:
        return s
    pat = re.compile(rf"\s*([{re.escape(allow_chars)}])\s+")
    # compress spaces on the right; run twice to catch "R  &   D"
    s = pat.sub(r"\1", s)
    s = pat.sub(r"\1", s)
    # also swallow spaces on the left side
    pat_left = re.compile(rf"\s+([{re.escape(allow_chars)}])")
    return pat_left.sub(r"\1", s)


def normalize_acronym_key(surface: str, allow_chars: str, dotted_mode: str) -> str:
    """
    Canonical form for acronym keys:
      - NFKC + fold dash/apostrophes
      - dotted policy: 'strip' removes '.', 'preserve' keeps them
      - swallow spaces around allowed internal separators (&-/.) only
    NO lowercasing here; upstream chooses case policy.
    """
    s = canonicalize(surface)
    if dotted_mode == "strip":
        s = s.replace(".", "")
    s = _swallow_spaces_around_allowed(s, allow_chars)
    return s



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


# Last Proper-Noun chunk, e.g. "North American Saxophone Alliance"
_LAST_PROPER_CHUNK = re.compile(r"([A-Z][\w’'-]+(?:\s+[A-Z][\w’'-]+){1,})$")


def tighten_label(s: str) -> str:
    s = normalize_definition(s)

    # drop leading connectors (run twice to be safe)
    for _ in range(2):
        s = LEADING_CONNECTORS.sub("", s).strip()

    # if we have a trailing proper-noun chunk, use it
    m = _LAST_PROPER_CHUNK.search(s)
    if m:
        chunk = m.group(1)
        return ARTICLE.sub("", chunk).strip()

    # else: remove leading article only
    s = ARTICLE.sub("", s).strip()

    # common “X is/means/stands for Y” patterns → keep the RHS
    for splitter in (" stands for ", " means ", " is ", " are "):
        parts = s.split(splitter, 1)
        if len(parts) == 2:
            return parts[1].strip()

    # fallback: return as-is (already normalized)
    return s
