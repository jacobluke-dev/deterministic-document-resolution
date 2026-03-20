import re
from typing import Iterable

from plainera_unacronym.nlp.common.types import AcronymMeaning, ExtractedDefinition
from plainera_unacronym.nlp.extraction.acronyms.core.defs import dedupe_defs
from plainera_unacronym.nlp.extraction.acronyms.core.normalise import tighten_label


def _slug(s: str) -> str:
    """Convert a string into a lowercase ASCII-ish slug.

    The slug keeps only `[a-z0-9]` and replaces other runs with `_`, then trims `_`.
    If the result is empty, returns `"x"`.

    Args:
        s: Input string.

    Returns:
        A slug string suitable for building stable sense IDs.
    """
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "x"


def build_senses(defs: Iterable[ExtractedDefinition]) -> dict[str, list[AcronymMeaning]]:
    """Build `AcronmMeaning` objects from definition matches.

    Definitions are de-duplicated first. Each definition becomes a "sense" keyed by:
    `"{acr.lower()}|{slug(tighten_label(definition))}"`. Multiple defs for the same sense
    are merged by appending spans and incrementing `support`.

    Args:
        defs (ExtractedDefinition): Iterable of definition-like objects with attributes:
            `acronym`, `definition`, `def_start`, `def_end`.

    Returns:
        Mapping `{ACRONYM: [AcronmMeaning, ...]}` where keys are uppercased acronyms.
    """
    senses_by: dict[str, dict[str, AcronymMeaning]] = {}

    for d in dedupe_defs(list(defs)):
        acr = d.acronym.upper()
        label = tighten_label(d.definition)
        sid = f"{acr.lower()}|{_slug(label)}"
        by_label = senses_by.setdefault(acr, {})

        sense = by_label.get(sid)
        if sense is None:
            sense = AcronymMeaning(
                acronym=acr,
                definition=label,
                sense_id=sid,
                def_spans=[],
                support=0,
                sense_confidence=d.definition_confidence,
            )
            by_label[sid] = sense

        sense.def_spans.append((d.def_start, d.def_end))
        sense.support += 1
        # important when multiple defs collapse to same sid across pipeline:
        if d.definition_confidence > sense.sense_confidence:
            sense.sense_confidence = d.definition_confidence

    return {acr: list(by.values()) for acr, by in senses_by.items()}
