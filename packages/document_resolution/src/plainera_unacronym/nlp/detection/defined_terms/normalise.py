import re

from plainera_unacronym.nlp.common.constants_regex import CANON_TABLE_DEFAULT

_WS_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^\w\s-]")


def normalize_defined_term_key(term: str) -> str:
    """Normalise a defined term into a stable lookup key.

    Args:
        term (str): term to normalise

    Returns:
        str: normalised term string

    Rules:
      - strip surrounding whitespace
      - collapse internal whitespace
      - remove surrounding quotes/punctuation noise
      - lowercase
      - preserve semantically meaningful bridge words
      - convert spaces/hyphens to underscores
    """
    value = term.translate(CANON_TABLE_DEFAULT)
    value = value.strip().strip("\"'")
    value = _NON_WORD_RE.sub("", value)
    value = _WS_RE.sub(" ", value).strip().lower()
    value = value.replace("-", " ")
    value = _WS_RE.sub("_", value)
    return value
