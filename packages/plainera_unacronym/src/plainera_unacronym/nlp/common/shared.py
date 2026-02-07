import re
import unicodedata

from plainera_unacronym.nlp.common.config import TRAILING_PUNCT, CANON_TABLE


def has_paren_definition(text: str, end: int, max_chars: int = 80) -> bool:
    """Return whether a parenthetical definition follows immediately after a token.

    Starting at `end` (the character index directly after a candidate token), this
    checks for optional whitespace, then a '(' and scans up to `max_chars` characters
    inside the parentheses looking for a closing ')'. It counts ASCII alphabetic
    characters within the parentheses and returns True only if a closing parenthesis
    is found within the limit and at least 5 ASCII letters are present.

    Args:
        text: The full source text being analyzed.
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
    """Canonicalises text using Unicode NFKC normalisation and look-alike folding.

    This function normalises `s` using Unicode NFKC to collapse compatibility
    characters (e.g., fullwidth forms) and then translates common look-alikes for:
      - apostrophes/quotes (various Unicode apostrophe-like code points) -> "'"
      - dashes (en dash/em dash) -> "-"

    The translation is performed via `CANON_TABLE` for speed.

    Args:
        s: Input string to canonicalise.

    Returns:
        A canonicalised string with NFKC applied and apostrophe/dash variants folded.
    """
    return unicodedata.normalize("NFKC", s).translate(CANON_TABLE)


def strip_trailing_punct_str(s: str) -> str:
    """Strips trailing punctuation and trailing whitespace from a string.

    Removes one or more characters from the end of `s` that match `TRAILING_PUNCT`,
    which is configured to include:
      - punctuation characters in `TRAILING_PUNCT_CHARS`
      - any trailing whitespace

    This is intended for lightweight token cleanup where punctuation has been attached
    to a token boundary (e.g., "RNA," -> "RNA", "Unit) " -> "Unit").

    Args:
        s: Input string.

    Returns:
        The input string with trailing punctuation/whitespace removed.
    """
    return re.sub(TRAILING_PUNCT, "", s)


def _swallow_spaces_around_allowed(s: str, allow_chars: str) -> str:
    """Collapses whitespace around “allowed” connector characters.

    This is used to normalise token shapes where certain connector characters are
    considered semantically significant (e.g., '&', '/', '+') and should not carry
    surrounding spaces.

    Behaviour:
      - If `allow_chars` is empty, returns `s` unchanged.
      - Removes whitespace to the right of an allowed char (run twice to handle
        multiple-space cases like "R  &   D").
      - Removes whitespace to the left of an allowed char.

    Example:
        "R & D"  -> "R&D"
        "A  /  B" -> "A/B"

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

    Steps:
      1) Unicode NFKC normalisation plus folding of look-alike apostrophes/dashes
         via `canonicalize()`.
      2) Dotted policy:
           - "strip": remove '.' characters
           - "preserve": keep '.' characters
      3) Collapse whitespace around allowed connector characters only
         (e.g., "R & D" -> "R&D") using `_swallow_spaces_around_allowed()`.

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
    """Collapses runs of whitespace into single spaces and trims ends.

    Replaces one or more whitespace characters (spaces, tabs, newlines, etc.) with a
    single ASCII space and strips leading/trailing whitespace.

    Args:
        s: Input string.

    Returns:
        Normalised string with collapsed internal whitespace and no leading/trailing
        whitespace.
    """
    return re.sub(r"\s+", " ", s).strip()


def has_letter(s: str) -> bool:
    """True if the string contains any Unicode letter.

    Args:
      s (str): String to check.

    Returns:
      bool: True if any character in ``s`` satisfies ``str.isalpha()``; else False.
    """
    return any(ch.isalpha() for ch in s)
