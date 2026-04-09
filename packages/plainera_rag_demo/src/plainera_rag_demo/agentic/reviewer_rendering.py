from typing import Any


def summarize_acronyms(value: Any) -> list[dict[str, Any]]:
    """Summarise acronym bindings for reviewer use."""
    if not isinstance(value, list):
        return []

    summaries: list[dict[str, Any]] = []

    for item in value:
        if not isinstance(item, dict):
            continue

        selected = item.get("selected")
        definition = None
        reason = None

        if isinstance(selected, dict):
            raw_definition = selected.get("definition")
            if isinstance(raw_definition, str):
                definition = raw_definition
            raw_reason = selected.get("reason")
            if isinstance(raw_reason, str):
                reason = raw_reason

        if definition is None:
            definitions = item.get("definitions")
            if isinstance(definitions, list) and definitions:
                first = definitions[0]
                if isinstance(first, dict):
                    raw_text = first.get("text")
                    if isinstance(raw_text, str):
                        definition = raw_text

        summaries.append(
            {
                "acronym": item.get("acronym"),
                "selected_definition": definition,
                "selection_reason": reason,
                "conflict": item.get("conflict"),
                "source_ref": first_candidate_source_ref(item),
            }
        )

    return summaries


def summarize_defined_terms(value: Any) -> list[dict[str, Any]]:
    """Summarise defined-term bindings for reviewer use."""
    if not isinstance(value, list):
        return []

    summaries: list[dict[str, Any]] = []

    for item in value:
        if not isinstance(item, dict):
            continue

        chosen_definition = None
        chosen_span = item.get("chosen_definition_span")
        if isinstance(chosen_span, dict):
            text = chosen_span.get("text")
            if isinstance(text, str):
                chosen_definition = text

        summaries.append(
            {
                "surface": item.get("surface"),
                "normalized_key": item.get("normalized_key"),
                "chosen_definition_text": chosen_definition,
                "resolution_method": item.get("resolution_method"),
                "resolved": item.get("resolved"),
            }
        )

    return summaries


def summarize_structural_references(value: Any) -> list[dict[str, Any]]:
    """Summarise structural references for reviewer use."""
    if not isinstance(value, list):
        return []

    summaries: list[dict[str, Any]] = []

    for item in value:
        if not isinstance(item, dict):
            continue

        reference_span = item.get("reference_span")
        target_span = item.get("target_span")

        summaries.append(
            {
                "reference_span": reference_span if isinstance(reference_span, dict) else None,
                "target_span": target_span if isinstance(target_span, dict) else None,
                "match_strategy": item.get("match_strategy"),
                "resolved": item.get("resolved"),
            }
        )

    return summaries


def summarize_grounding_payload(grounding_payload: dict[str, Any]) -> dict[str, Any]:
    """Reduce raw resolve output to the small subset the reviewer actually needs."""
    return {
        "acronyms": summarize_acronyms(
            grounding_payload.get("acronyms")
        ),
        "defined_terms": summarize_defined_terms(
            grounding_payload.get("defined_terms")
        ),
        "structural_references": summarize_structural_references(
            grounding_payload.get("structural_references")
        ),
    }


def first_candidate_source_ref(item: dict[str, Any]) -> str | None:
    """Return the first candidate source ref when available."""
    candidates = item.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None

    first = candidates[0]
    if not isinstance(first, dict):
        return None

    value = first.get("source_ref")
    return value if isinstance(value, str) else None


def extract_ambiguity_indicators(grounding_payload: dict[str, Any]) -> dict[str, Any]:
    """Extract lightweight ambiguity markers from grounding payload."""
    acronyms = grounding_payload.get("acronyms", [])
    defined_terms = grounding_payload.get("defined_terms", [])
    structural_references = grounding_payload.get("structural_references", [])

    def _count_unresolved(items: Any) -> int:
        if not isinstance(items, list):
            return 0

        unresolved = 0
        for item in items:
            if not isinstance(item, dict):
                continue

            selected = item.get("selected")
            if isinstance(selected, dict):
                definition = selected.get("definition")
                reason = selected.get("reason")
                if reason == "unresolved" or not isinstance(definition, str) or not definition.strip():
                    unresolved += 1
                continue

            if item.get("resolution_method") == "unresolved" or item.get("resolved") is False:
                unresolved += 1

        return unresolved

    return {
        "acronym_count": len(acronyms) if isinstance(acronyms, list) else 0,
        "defined_term_count": len(defined_terms) if isinstance(defined_terms, list) else 0,
        "structural_reference_count": (
            len(structural_references) if isinstance(structural_references, list) else 0
        ),
        "unresolved_acronyms": _count_unresolved(acronyms),
        "unresolved_defined_terms": _count_unresolved(defined_terms),
        "unresolved_structural_references": _count_unresolved(structural_references),
    }
