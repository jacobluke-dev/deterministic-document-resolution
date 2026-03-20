from __future__ import annotations

from collections import defaultdict

from plainera_unacronym.nlp.extraction.defined_terms.types import TermDefinitionEntry, TermMeaning


def build_term_meaning_index(
    definition_entries: list[TermDefinitionEntry],
) -> tuple[dict[str, tuple[TermMeaning, ...]], dict[str, TermMeaning]]:
    """
    Build deterministic term meanings from extracted definition entries.

    Ordinals are assigned per normalized key in document order, so repeated
    introductions of the same term become:

        term|services|1
        term|services|2
        ...

    Args:
        definition_entries: Extracted term definition records.

    Returns:
        tuple[
            dict[str, tuple[TermMeaning, ...]],
            dict[str, TermMeaning],
        ]:
            - grouped meaning index keyed by normalized term key
            - flat meaning lookup keyed by meaning_id
    """
    grouped: dict[str, list[TermMeaning]] = defaultdict(list)
    flat: dict[str, TermMeaning] = {}
    ordinals_by_key: dict[str, int] = defaultdict(int)

    ordered_entries = sorted(
        definition_entries,
        key=lambda e: (e.intro_span[1], e.intro_span[2], e.normalized_key),
    )

    for entry in ordered_entries:
        ordinals_by_key[entry.normalized_key] += 1
        ordinal = ordinals_by_key[entry.normalized_key]
        meaning_id = f"term|{entry.normalized_key}|{ordinal}"

        meaning = TermMeaning(
            meaning_id=meaning_id,
            surface=entry.surface,
            normalized_key=entry.normalized_key,
            ordinal=ordinal,
            intro_span=entry.intro_span,
            definition_span=entry.definition_span,
            definition_text=entry.definition_text,
            intro_kind=entry.intro_kind,
            alias_target_span=entry.alias_target_span,
            alias_target_text=entry.alias_target_text,
            section_path=entry.section_path,
        )

        grouped[entry.normalized_key].append(meaning)
        flat[meaning_id] = meaning

    return (
        {k: tuple(v) for k, v in grouped.items()},
        flat,
    )
