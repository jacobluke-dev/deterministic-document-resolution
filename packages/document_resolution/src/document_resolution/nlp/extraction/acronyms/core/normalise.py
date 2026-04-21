import re

from document_resolution.nlp.common.constants_regex import ARTICLE, LEADING_CONNECTORS
from document_resolution.nlp.common.shared import canonicalize, collapse_ws, strip_trailing_punct_str


def normalize_definition(s: str) -> str:
    """Normalise a definition string for consistent UX/display.

    This is a *presentation-layer* normaliser used to make extracted long-forms
    stable and user-friendly without changing their semantic content.

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

    Args:
        def_str: Candidate definition string.

    Returns:
        A tightened, display-friendly definition label.
    """
    # 1. canonicalise apostrophes/dashes, collapse whitespace, trim trailing punctuation
    def_str = normalize_definition(def_str)

    # 2. Remove common leading connector phrases (e.g. "and", "which", "for")
    for _ in range(2):
        def_str = LEADING_CONNECTORS.sub("", def_str).strip()

    # 3. if we have a trailing proper-noun chunk, use it
    m = _LAST_PROPER_CHUNK.search(def_str)
    if m:
        chunk = m.group(1)
        return ARTICLE.sub("", chunk).strip()

    # 4. else: Otherwise, remove a single leading article ("the", "a", "an").
    def_str = ARTICLE.sub("", def_str).strip()

    # 5.  If the string contains “X is/means/stands for Y” patterns → keep the RHS
    for splitter in (" stands for ", " means ", " is ", " are "):
        parts = def_str.split(splitter, 1)
        if len(parts) == 2:
            return parts[1].strip()

    # 5b. fallback: return as-is (already normalized)
    return def_str
