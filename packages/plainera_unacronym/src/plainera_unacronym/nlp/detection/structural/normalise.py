import re

from plainera_unacronym.nlp.common.constants_regex import CANON_TABLE_DEFAULT

_WS_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^\w\s-]")


def normalize_structural_reference_key(kind: str, label: str) -> str:
    """
    Examples:
      - Schedule A  -> schedule_a
      - Section 4.2 -> section_4_2
      - Article III -> article_iii
    """
    kind_value = kind.translate(CANON_TABLE_DEFAULT).strip().lower()
    kind_value = _NON_WORD_RE.sub("", kind_value)
    kind_value = _WS_RE.sub("_", kind_value)

    label_value = label.translate(CANON_TABLE_DEFAULT).strip().lower()
    label_value = label_value.replace(".", "_")
    label_value = _NON_WORD_RE.sub("", label_value)
    label_value = _WS_RE.sub("_", label_value)

    return f"{kind_value}_{label_value}"
