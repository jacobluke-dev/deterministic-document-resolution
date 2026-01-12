from plainera_unacronym.nlp.common.types import ExtractedDefinition
from plainera_unacronym.nlp.extraction.matchers.helper_patterns import (
    find_parenthetical_longform_after_acr,
    find_parenthetical_longform_before_acr,
)
from plainera_unacronym.nlp.extraction.matchers.tighten import tighten_label_by_acronym


def harvest_defs_all(text: str, occs, cfg) -> list[ExtractedDefinition]:
    out: list[ExtractedDefinition] = []
    win = getattr(cfg, "window_chars", 320)

    for o in occs:
        a0, a1 = o.start_offset, o.end_offset
        L = max(0, a0 - win)
        R = min(len(text), a1 + win)
        snippet = text[L:R]
        rel_a0, rel_a1 = a0 - L, a1 - L

        pre = snippet[: min(len(snippet), rel_a1 + 1)]

        for m in find_parenthetical_longform_before_acr(pre, o.acronym, cfg):
            ds, de = L + m.def_start, L + m.def_end
            out.append(
                ExtractedDefinition(
                    acronym=o.acronym,
                    definition=tighten_label_by_acronym(m.definition, o.acronym.upper()),
                    source="in_text",
                    confidence=0.95,
                    acr_start=a0,
                    acr_end=a1,
                    def_start=ds,
                    def_end=de,
                    original_definition=text[ds:de],  # <-- raw slice from full text
                )
            )

        # ACR (Long form) after
        right = snippet[rel_a1:]
        for m in find_parenthetical_longform_after_acr(right, cfg=cfg, acr=o.acronym):

            ds, de = (L + rel_a1) + m.def_start, (L + rel_a1) + m.def_end
            out.append(
                ExtractedDefinition(
                    acronym=o.acronym,
                    definition=tighten_label_by_acronym(m.definition, o.acronym.upper()),
                    source="in_text",
                    confidence=0.95,
                    acr_start=a0,
                    acr_end=a1,
                    def_start=ds,
                    def_end=de,
                    original_definition=text[ds:de],
                )
            )

    return out
