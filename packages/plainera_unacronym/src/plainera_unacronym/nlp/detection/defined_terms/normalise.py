import re


_WS_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^\w\s-]")


def normalize_defined_term_key(term: str) -> str:
    """Normalise a defined term into a stable lookup key.

    Rules:
      - strip surrounding whitespace
      - collapse internal whitespace
      - remove surrounding quotes/punctuation noise
      - lowercase
      - preserve semantically meaningful bridge words
      - convert spaces/hyphens to underscores
    """
    value = term.strip().strip("\"'“”‘’")
    value = _NON_WORD_RE.sub("", value)
    value = _WS_RE.sub(" ", value).strip().lower()
    value = value.replace("-", " ")
    value = _WS_RE.sub("_", value)
    return value
