
from typing import Dict, List, Optional, Tuple, Set

from plainera_unacronym.nlp.common.shared import tighten_label
from plainera_unacronym.nlp.common.types import InTextPick, ExtractedDefinition
from plainera_unacronym.nlp.extraction.tighten import tighten_label_by_acronym


def defs_from_picks(text: str, picks: Dict[str, Optional[InTextPick]]) -> List[ExtractedDefinition]:
    out: List[ExtractedDefinition] = []
    for _, pick in picks.items():
        if pick is None:
            continue
        a0, a1 = pick.acr_span
        acr_surface = text[a0:a1]
        out.append(
            ExtractedDefinition(
                acronym=acr_surface.upper(),
                definition=tighten_label_by_acronym(pick.definition, acr_surface.upper()),
                source="in_text",
                confidence=pick.confidence,
                acr_start=a0, acr_end=a1,
                def_start=pick.def_span[0], def_end=pick.def_span[1],
                original_definition=pick.original_definition,
            )
        )
    return out


def _sense_key(acr: str, label: str) -> Tuple[str, str]:
    return acr.upper(), tighten_label(label).lower()


def dedupe_defs(defs: List[ExtractedDefinition]) -> List[ExtractedDefinition]:
    seen: Set[Tuple[str, str]] = set()
    out: List[ExtractedDefinition] = []
    for d in defs:
        k = _sense_key(d.acronym, d.definition)
        if k in seen:
            continue
        seen.add(k)
        # keep d.definition as-is (already tightened upstream where needed)
        out.append(d)
    return out
