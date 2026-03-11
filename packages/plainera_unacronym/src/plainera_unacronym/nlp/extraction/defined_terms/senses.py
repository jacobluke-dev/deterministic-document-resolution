from __future__ import annotations

from collections import defaultdict

from plainera_unacronym.nlp.extraction.defined_terms.types import TermDefinitionEntry, TermSense


def build_term_sense_index(
    definition_entries: list[TermDefinitionEntry],
) -> tuple[dict[str, tuple[TermSense, ...]], dict[str, TermSense]]:
    """
    Build deterministic term senses from extracted definition entries.

    Ordinals are assigned per normalized key in document order, so repeated
    introductions of the same term become:

        term|services|1
        term|services|2
        ...

    Args:
        definition_entries: Extracted term definition records.

    Returns:
        tuple[
            dict[str, tuple[TermSense, ...]],
            dict[str, TermSense],
        ]:
            - grouped sense index keyed by normalized term key
            - flat sense lookup keyed by sense_id
    """
    grouped: dict[str, list[TermSense]] = defaultdict(list)
    flat: dict[str, TermSense] = {}
    ordinals_by_key: dict[str, int] = defaultdict(int)

    ordered_entries = sorted(
        definition_entries,
        key=lambda e: (e.intro_span[1], e.intro_span[2], e.normalized_key),
    )

    for entry in ordered_entries:
        ordinals_by_key[entry.normalized_key] += 1
        ordinal = ordinals_by_key[entry.normalized_key]
        sense_id = f"term|{entry.normalized_key}|{ordinal}"

        sense = TermSense(
            sense_id=sense_id,
            surface=entry.surface,
            normalized_key=entry.normalized_key,
            ordinal=ordinal,
            intro_span=entry.intro_span,
            definition_span=entry.definition_span,
            definition_text=entry.definition_text,
            intro_kind=entry.intro_kind,
            section_path=entry.section_path,
        )

        grouped[entry.normalized_key].append(sense)
        flat[sense_id] = sense

    return (
        {k: tuple(v) for k, v in grouped.items()},
        flat,
    )
