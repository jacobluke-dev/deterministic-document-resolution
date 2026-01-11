import re

from plainera_unacronym.nlp.common.shared import tighten_label
from plainera_unacronym.nlp.common.types import AcronymSense
from plainera_unacronym.nlp.extraction.core.defs import dedupe_defs


def _slug(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "x"


def build_senses(defs) -> dict[str, list[AcronymSense]]:
    senses_by: dict[str, dict[str, AcronymSense]] = {}
    for d in dedupe_defs(list(defs)):
        acr = d.acronym.upper()
        label = tighten_label(d.definition)
        sid = f"{acr.lower()}|{_slug(label)}"
        by_label = senses_by.setdefault(acr, {})
        sense = by_label.get(sid)
        if not sense:
            sense = AcronymSense(acronym=acr, definition=label, sense_id=sid, def_spans=[], support=0)
            by_label[sid] = sense
        sense.def_spans.append((d.def_start, d.def_end))
        sense.support += 1
    # flatten
    return {acr: list(d.values()) for acr, d in senses_by.items()}
