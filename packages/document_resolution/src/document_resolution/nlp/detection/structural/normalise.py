import re

from document_resolution.nlp.common.constants_regex import CANON_TABLE_DEFAULT

_WS_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^\w\s-]")


def normalize_structural_reference_key(kind: str, label: str) -> str:
    """Normalise a structural reference into a stable lookup key.

    Args:
        kind: Structural reference kind to normalise, for example ``"Section"``
            or ``"Schedule"``.
        label: Structural reference label to normalise, for example ``"4.2"``,
            ``"A"``, or ``"III"``.

    Returns:
        A deterministic normalised key combining the cleaned structural kind and
        cleaned label, for example ``"section_4_2"`` or ``"schedule_a"``.

    Rules:
      - translate canonical punctuation variants using ``CANON_TABLE_DEFAULT``
      - strip surrounding whitespace
      - lowercase all content
      - remove non-word punctuation from kind and label
      - convert decimal points in labels to underscores
      - collapse whitespace runs to underscores
    """
    kind_value = kind.translate(CANON_TABLE_DEFAULT).strip().lower()
    kind_value = _NON_WORD_RE.sub("", kind_value)
    kind_value = _WS_RE.sub("_", kind_value)

    label_value = label.translate(CANON_TABLE_DEFAULT).strip().lower()
    label_value = label_value.replace(".", "_")
    label_value = _NON_WORD_RE.sub("", label_value)
    label_value = _WS_RE.sub("_", label_value)

    return f"{kind_value}_{label_value}"
