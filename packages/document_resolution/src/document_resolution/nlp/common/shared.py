import re
import unicodedata

from document_resolution.nlp.common.config import CANON_TABLE, TRAILING_PUNCT


def has_paren_definition(text: str, end: int, max_chars: int = 80) -> bool:
    """Return whether a likely parenthetical definition follows a token.

    Skips whitespace after `end`, requires an opening parenthesis, and scans up to
    `max_chars` characters for a closing parenthesis containing at least 5 ASCII
    letters.

    Args:
        text: The full source text.
        end: The index in `text` immediately after the token to test.
        max_chars: Maximum number of characters to scan inside the parentheses before
            giving up. The closing ')' must appear within this window.

    Returns:
        True if a parenthetical definition is detected within the limit and contains
        at least 5 ASCII letters; otherwise False.
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
    """Apply unicode normalisation and fold apostrophe/dash variants
    Args:
        s: Input string to canonicalise.

    Returns:
        A canonicalised string with NFKC applied and apostrophe/dash variants folded.
    """
    return unicodedata.normalize("NFKC", s).translate(CANON_TABLE)


def strip_trailing_punct_str(s: str) -> str:
    """Remove trailing punctuation and whitespace.
     (e.g., "RNA," -> "RNA", "Unit) " -> "Unit").

    Args:
        s: Input string.

    Returns:
        The input string with trailing punctuation/whitespace removed.
    """
    return re.sub(TRAILING_PUNCT, "", s)


def _swallow_spaces_around_allowed(s: str, allow_chars: str) -> str:
    """Remove whitespace around allowed connector characters.

    For example, normalises `R & D` to `R&D`.

    Args:
        s: Input string to normalise.
        allow_chars: Characters to treat as “connectors” whose surrounding whitespace
            should be removed (interpreted as a literal set; regex-escaped).

    Returns:
        A string with whitespace around connector characters removed.
    """
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
    """Normalises an acronym surface form into a canonical key.

    This produces a stable key used for grouping occurrences. The normalisation is
    intentionally minimal and does not apply case folding.

    Args:
        surface: The raw acronym string as it appears in text.
        allow_chars: Connector characters to treat as “internal” (e.g., "&/-+.").
        dotted_mode: Either "strip" or "preserve".

    Returns:
        The canonical acronym key.
    """
    s = canonicalize(surface)
    if dotted_mode == "strip":
        s = s.replace(".", "")
    s = _swallow_spaces_around_allowed(s, allow_chars)
    return s


def collapse_ws(s: str) -> str:
    """
    Collapses runs of whitespace into single spaces and trims ends.
    """
    return re.sub(r"\s+", " ", s).strip()


def has_letter(s: str) -> bool:
    return any(ch.isalpha() for ch in s)
