import re
from collections.abc import Iterable

from document_resolution.nlp.common.types import AcronymMeaning, ExtractedDefinition
from document_resolution.nlp.extraction.acronyms.core.defs import dedupe_defs
from document_resolution.nlp.extraction.acronyms.core.normalise import tighten_label


def _slug(s: str) -> str:
    """Convert a string into a lowercase ASCII-ish slug.

    Args:
        s: Input string.

    Returns:
        A slug string suitable for building stable meaning IDs.
    """
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "x"


def build_meanings(defs: Iterable[ExtractedDefinition]) -> dict[str, list[AcronymMeaning]]:
    """Build `AcronymMeaning` objects from definition matches.

    Args:
        defs (ExtractedDefinition): Iterable of definition-like objects with attributes:
            `acronym`, `definition`, `def_start`, `def_end`.

    Returns:
        Mapping `{ACRONYM: [AcronmMeaning, ...]}` where keys are uppercased acronyms.
    """
    meamings_by: dict[str, dict[str, AcronymMeaning]] = {}

    for d in dedupe_defs(list(defs)):
        acr = d.acronym.upper()
        label = tighten_label(d.definition)
        sid = f"{acr.lower()}|{_slug(label)}"
        by_label = meamings_by.setdefault(acr, {})

        meaning = by_label.get(sid)
        if meaning is None:
            meaning = AcronymMeaning(
                acronym=acr,
                definition=label,
                meaning_id=sid,
                def_spans=[],
                support=0,
                meaning_confidence=d.definition_confidence,
            )
            by_label[sid] = meaning

        meaning.def_spans.append((d.def_start, d.def_end))
        meaning.support += 1
        # important when multiple defs collapse to same sid across pipeline:
        if d.definition_confidence > meaning.meaning_confidence:
            meaning.meaning_confidence = d.definition_confidence

    return {acr: list(by.values()) for acr, by in meamings_by.items()}
