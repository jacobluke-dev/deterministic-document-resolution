from plainera_unacronym.nlp.common.types import ExtractedDefinition
from plainera_unacronym.nlp.extraction import ExtractionConfig
from plainera_unacronym.nlp.extraction.acronyms.engine.confidence import base_conf_for
from plainera_unacronym.nlp.extraction.acronyms.matchers.defs import find_parenthetical_longform_before_acr, \
    find_parenthetical_longform_after_acr
from plainera_unacronym.nlp.extraction.acronyms.matchers.tighten import tighten_label_by_acronym


def extract_defs_all_occurrences(text: str, occs, cfg: ExtractionConfig) -> list[ExtractedDefinition]:
    """
    Extract parenthetical acronym definitions around detected occurrences.

    For each occurrence, this scans a fixed character window around the acronym and runs two
    matchers:
      - Long form before acronym: "Portable Document Format (PDF)"
      - Long form after acronym:  "PDF (Portable Document Format)"

    Each matcher returns local (snippet-relative) spans; this function converts them back to
    absolute spans in `text` and emits `ExtractedDefinition` objects with:
      - `acronym` copied from the occurrence,
      - `definition` tightened via `tighten_label_by_acronym(..., acronym.upper())`,
      - `original_definition` taken as the raw slice `text[def_start:def_end]`.

    Args:
        text: Full source text.
        occs: Iterable of occurrence-like objects exposing:
            `acronym` (str), `start_offset` (int), `end_offset` (int, exclusive).
        cfg: Config object; may define `window_chars` (int, default 320) used to bound the scan.

    Returns:
        A list of `ExtractedDefinition` objects (possibly empty). Results are appended in the
        same order as `occs`, with any "before" match(es) emitted before any "after" match(es)
        for the same occurrence.

    Raises:
        AttributeError: If occurrences do not provide the expected attributes.
        IndexError: If a matcher returns invalid span indices outside the snippet bounds.
    """
    out: list[ExtractedDefinition] = []
    win = getattr(cfg, "window_chars", 320)
    SRC = "all_occ_scan_parenthetical"
    base = base_conf_for(cfg, source=SRC, default=0.8)

    for o in occs:
        a0, a1 = o.start_offset, o.end_offset
        L = max(0, a0 - win)
        R = min(len(text), a1 + win)
        snippet = text[L:R]
        _, rel_a1 = a0 - L, a1 - L

        pre = snippet[: min(len(snippet), rel_a1 + 1)]

        for m in find_parenthetical_longform_before_acr(pre, o.acronym, cfg):
            ds, de = L + m.def_start, L + m.def_end
            out.append(
                ExtractedDefinition(
                    acronym=o.acronym,
                    definition=tighten_label_by_acronym(m.definition, o.acronym.upper()),
                    source=SRC,
                    definition_confidence=base,
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
                    source=SRC,
                    definition_confidence=base,
                    acr_start=a0,
                    acr_end=a1,
                    def_start=ds,
                    def_end=de,
                    original_definition=text[ds:de],
                )
            )

    return out
