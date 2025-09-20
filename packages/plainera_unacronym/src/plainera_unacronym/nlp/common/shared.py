import re
import unicodedata

from plainera_unacronym.nlp.common.config import CANON_TABLE, TRAILING_PUNCT

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

def _canonicalize(s: str) -> str:
    # NFKC normalisation + map look-alikes (apostrophes, dashes)
    return unicodedata.normalize("NFKC", s).translate(CANON_TABLE)

def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _strip_trailing_punct(s: str) -> str:
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
    s = _canonicalize(surface)
    if dotted_mode == "strip":
        s = s.replace(".", "")
    s = _swallow_spaces_around_allowed(s, allow_chars)
    return s

def normalize_definition(s: str) -> str:
    """
    UX/display normalisation for definitions:
      - NFKC + fold dash/apostrophes
      - collapse whitespace
      - strip trailing punctuation
    """
    return _strip_trailing_punct(_collapse_ws(_canonicalize(s)))
