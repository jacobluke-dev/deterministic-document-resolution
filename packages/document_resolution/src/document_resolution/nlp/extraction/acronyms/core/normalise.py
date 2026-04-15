import re

from document_resolution.nlp.common.constants_regex import ARTICLE, LEADING_CONNECTORS
from document_resolution.nlp.common.shared import canonicalize, collapse_ws, strip_trailing_punct_str


def normalize_definition(s: str) -> str:
    """Normalise a definition string for consistent UX/display.

    This is a *presentation-layer* normaliser used to make extracted long-forms
    stable and user-friendly without changing their semantic content.

    It performs three steps:

    1) Canonicalisation (`canonicalize`)
       - Applies Unicode NFKC normalisation.
       - Folds common dash variants (en/em dashes) to ASCII "-".
       - Folds common apostrophe variants (curly quotes, primes) to ASCII "'".

    2) Whitespace normalisation (`collapse_ws`)
       - Collapses any run of whitespace (spaces, tabs, newlines) to a single
         ASCII space.
       - Trims leading and trailing whitespace.

    3) Trailing punctuation trimming (`strip_trailing_punct_str`)
       - Removes trailing punctuation characters (e.g. ",.;:)]}»”'\"") and any
         trailing whitespace.

    This function is intentionally conservative: it does not lowercase, remove
    internal punctuation, or rewrite wording. It is designed for deterministic
    display output and easier deduplication across extraction paths.

    Args:
        s: Raw definition text (may contain Unicode punctuation and messy spacing).

    Returns:
        A normalised definition string suitable for display and comparison.
    """
    return strip_trailing_punct_str(collapse_ws(canonicalize(s)))


# Last Proper-Noun chunk, e.g. "North American Saxophone Alliance"
_LAST_PROPER_CHUNK = re.compile(r"([A-Z][\w’'-]+(?:\s+[A-Z][\w’'-]+){1,})$")


def tighten_label(def_str: str) -> str:
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
        def_str: Candidate definition string.

    Returns:
        A tightened, display-friendly definition label.
    """
    def_str = normalize_definition(def_str)

    # drop leading connectors (run twice to be safe)
    for _ in range(2):
        def_str = LEADING_CONNECTORS.sub("", def_str).strip()

    # if we have a trailing proper-noun chunk, use it
    m = _LAST_PROPER_CHUNK.search(def_str)
    if m:
        chunk = m.group(1)
        return ARTICLE.sub("", chunk).strip()

    # else: remove leading article only
    def_str = ARTICLE.sub("", def_str).strip()

    # common “X is/means/stands for Y” patterns → keep the RHS
    for splitter in (" stands for ", " means ", " is ", " are "):
        parts = def_str.split(splitter, 1)
        if len(parts) == 2:
            return parts[1].strip()

    # fallback: return as-is (already normalized)
    return def_str


def has_digit(s: str) -> bool:
    """True if the string contains any number.

    Args:
      s (str): String to check.

    Returns:
      bool: True if any character in ``s`` satisfies ``str.isalpha()``; else False.
    """
    return any(ch.isdigit() for ch in s)
