from __future__ import annotations


def meaning_by_id(state, meaning_id: str):
    return state.tier_1.meaning_index[meaning_id]

def resolution_key(r) -> str | None:
    if hasattr(r, "normalized_key"):
        return r.normalized_key
    if hasattr(r, "term_key"):
        return r.term_key
    if hasattr(r, "key"):
        return r.key

    occ = getattr(r, "occurrence", None)
    if occ is not None and hasattr(occ, "normalized_key"):
        return occ.normalized_key

    return None


def chosen_meaning_ids_for_key(extr, key: str) -> list[str]:
    return [
        r.chosen_meaning_id
        for r in resolutions_for_key(extr, key)
        if getattr(r, "chosen_meaning_id", None) is not None
    ]


def resolutions_for_key(extr, key: str):
    return [r for r in extr.term_resolutions if resolution_key(r) == key]


def meaning_text_by_id(state) -> dict[str, str]:
    out: dict[str, str] = {}

    for meaning_id, meaning in state.tier_1.meaning_index.items():
        definition_text = getattr(meaning, "definition_text", None)

        if not definition_text:
            for entry in state.definition_entries:
                if getattr(entry, "meaning_id", None) == meaning_id:
                    definition_text = getattr(entry, "definition_text", None)
                    break

        out[meaning_id] = (definition_text or "").lower()

    return out
