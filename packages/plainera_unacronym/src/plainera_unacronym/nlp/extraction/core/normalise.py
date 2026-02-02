import re

from plainera_unacronym.nlp.common.constants_regex import ARTICLE, LEADING_CONNECTORS
from plainera_unacronym.nlp.common.shared import canonicalize, collapse_ws, strip_trailing_punct_str


def normalize_definition(s: str) -> str:
    """
    UX/display normalisation for definitions:
      - NFKC + fold dash/apostrophes
      - collapse whitespace
      - strip trailing punctuation
    """
    return strip_trailing_punct_str(collapse_ws(canonicalize(s)))

# Last Proper-Noun chunk, e.g. "North American Saxophone Alliance"
_LAST_PROPER_CHUNK = re.compile(r"([A-Z][\w’'-]+(?:\s+[A-Z][\w’'-]+){1,})$")


def tighten_label(s: str) -> str:
    """Normalise and tighten a candidate definition label for display.

        This is a *display/UX* helper that tries to reduce noisy surrounding text
        while keeping the most meaningful phrase.

        Pipeline:
          1) `normalize_definition(s)` (canonicalise apostrophes/dashes, collapse
             whitespace, trim trailing punctuation).
          2) Remove common leading connector phrases (e.g. "and", "which", "for")
             up to two times to handle stacked connectors.
          3) If the string ends with a multi-word Proper-Noun chunk (TitleCase /
             ALLCAPS-ish words), return that trailing chunk (with a leading article removed).
          4) Otherwise, remove a single leading article ("the", "a", "an").
          5) If the string contains " stands for ", " means ", " is ", or " are ",
             return the RHS (right-hand side) as the tightened label.
          6) Fallback: return the (already normalised) string.

        Args:
            s: Candidate definition string.

        Returns:
            A tightened, display-friendly definition label.
    """
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


def has_letters(s: str) -> bool:
    """True if the string contains any Unicode letter.

    Args:
      s (str): String to check.

    Returns:
      bool: True if any character in ``s`` satisfies ``str.isalpha()``; else False.
    """
    return any(ch.isalpha() for ch in s)
