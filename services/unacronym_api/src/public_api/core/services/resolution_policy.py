from __future__ import annotations

from typing import Any

from public_api.db.repos.glossary_repo import GlossaryRepository
from public_api.schemas.resolve import ResolutionMode, ResolveOptions


def norm_definition(text: str) -> str:
    """Normalise a definition string for de-duplication and comparison."""
    return text.strip().rstrip(" .;,:").casefold()


def candidate_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    """Return a deterministic sort key for resolution candidates."""
    return (
        0 if candidate.get("provenance") == "document" else 1,
        str(candidate.get("definition") or "").casefold(),
        "" if candidate.get("domain") is None else str(candidate["domain"]).casefold(),
        str(candidate.get("source_ref") or ""),
    )


def build_document_candidates(
    definitions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[str]]:
    """Build resolution candidates derived from in-document definitions."""
    candidates: list[dict[str, Any]] = []
    seen_doc_defs: set[str] = set()

    for definition in definitions:
        text = str(definition.get("text") or "").strip()
        if not text:
            continue

        start = definition.get("start")
        end = definition.get("end")
        source_ref = None
        if isinstance(start, int) and isinstance(end, int):
            source_ref = f"text_span:{start}-{end}"

        candidates.append(
            {
                "domain": None,
                "definition": text,
                "score": 0.0,
                "provenance": "document",
                "source_ref": source_ref,
            }
        )
        seen_doc_defs.add(norm_definition(text))

    return candidates, seen_doc_defs


def build_glossary_candidates(
    meanings: list[dict[str, Any]],
    seen_doc_defs: set[str],
) -> tuple[list[dict[str, Any]], int]:
    """Build glossary-derived resolution candidates excluding document duplicates."""
    inactive_count = sum(1 for meaning in meanings if not bool(meaning.get("is_active")))
    active_meanings = [meaning for meaning in meanings if bool(meaning.get("is_active"))]

    candidates: list[dict[str, Any]] = []

    for meaning in active_meanings:
        definition = str(meaning.get("definition") or "").strip()
        if not definition:
            continue

        if norm_definition(definition) in seen_doc_defs:
            continue

        meaning_id = meaning.get("meaning_id")
        source_ref = f"meaning:{meaning_id}" if meaning_id is not None else None

        candidates.append(
            {
                "domain": meaning.get("domain"),
                "definition": definition,
                "score": 0.0,
                "provenance": "glossary",
                "source_ref": source_ref,
            }
        )

    return candidates, inactive_count


def select_resolution_candidate(
    candidates: list[dict[str, Any]],
    inactive_count: int,
    resolution_mode: ResolutionMode,
) -> tuple[dict[str, Any] | None, str | None]:
    """Select the best resolution candidate and explain the selection reason."""
    document_candidates = [c for c in candidates if c.get("provenance") == "document"]
    glossary_candidates = [c for c in candidates if c.get("provenance") == "glossary"]

    if document_candidates:
        document_candidates.sort(key=candidate_sort_key)
        return dict(document_candidates[0]), "in_document_definition"

    if not glossary_candidates:
        return None, None

    glossary_candidates.sort(key=candidate_sort_key)

    if resolution_mode == ResolutionMode.STRICT:
        if len(glossary_candidates) == 1:
            return (
                dict(glossary_candidates[0]),
                "inactive_filtered" if inactive_count > 0 else "single_candidate",
            )
        return None, None

    if resolution_mode == ResolutionMode.FALLBACK_GENERAL:
        general_candidate = next(
            (c for c in glossary_candidates if str(c.get("domain") or "").casefold() == "general"),
            None,
        )
        if general_candidate is not None:
            return dict(general_candidate), "fallback_general"

        return dict(glossary_candidates[0]), "highest_score"

    if len(glossary_candidates) == 1:
        return (
            dict(glossary_candidates[0]),
            "inactive_filtered" if inactive_count > 0 else "single_candidate",
        )

    general_candidate = next(
        (c for c in glossary_candidates if str(c.get("domain") or "").casefold() == "general"),
        None,
    )
    if general_candidate is not None:
        return dict(general_candidate), "fallback_general"

    return dict(glossary_candidates[0]), "highest_score"


def order_candidates(
    candidates: list[dict[str, Any]],
    selected: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Order candidates deterministically and promote the selected candidate first."""
    if selected is None:
        return sorted(candidates, key=candidate_sort_key)

    def same_candidate(a: dict[str, Any], b: dict[str, Any]) -> bool:
        return (
            a.get("definition") == b.get("definition")
            and a.get("domain") == b.get("domain")
            and a.get("provenance") == b.get("provenance")
            and a.get("source_ref") == b.get("source_ref")
        )

    remaining: list[dict[str, Any]] = []
    for candidate in candidates:
        if same_candidate(candidate, selected):
            candidate["score"] = 1.0
        else:
            candidate["score"] = 0.0
            remaining.append(candidate)

    remaining.sort(key=candidate_sort_key)

    selected = dict(selected)
    selected["score"] = 1.0
    return [selected, *remaining]


def attach_resolution_metadata(
    *,
    blocks: list[dict[str, Any]],
    opts: ResolveOptions,
    resolution_mode: ResolutionMode,
    glossary_repo: GlossaryRepository,
) -> list[dict[str, Any]]:
    """Attach candidate, selection, and conflict metadata to acronym blocks."""
    max_k = int(opts.max_definitions_per_acronym)
    out: list[dict[str, Any]] = []

    for block in blocks:
        nb = dict(block)
        ac = str(nb.get("acronym") or "").strip()
        if not ac:
            out.append(nb)
            continue

        definitions = nb.get("definitions") or []
        meanings = glossary_repo.list_meanings(acronym=ac)

        doc_candidates, seen_doc_defs = build_document_candidates(definitions)
        glossary_candidates, inactive_count = build_glossary_candidates(meanings, seen_doc_defs)

        candidates = [*doc_candidates, *glossary_candidates]
        viable_count = len(candidates)

        selected, reason = select_resolution_candidate(
            candidates,
            inactive_count,
            resolution_mode,
        )
        ordered_candidates = order_candidates(candidates, selected)

        nb["candidates"] = ordered_candidates[:max_k] if max_k > 0 else []
        nb["selected"] = (
            {
                "domain": selected.get("domain"),
                "definition": selected.get("definition"),
                "reason": reason,
            }
            if selected is not None and reason is not None
            else None
        )
        nb["conflict"] = viable_count > 1
        nb["conflict_count"] = viable_count
        nb["selection"] = {
            "policy_used": None,
            "filtered_inactive_count": inactive_count,
        }
        out.append(nb)

    return out
