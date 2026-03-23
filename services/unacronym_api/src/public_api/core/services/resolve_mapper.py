from __future__ import annotations

from typing import Any, Protocol

from plainera_unacronym.nlp.common.types import AcronymDetectorResult, ExtractionResult

from public_api.db.repos.glossary_repo import GlossaryRepository
from public_api.schemas.resolve import ResolveOptions


class _SpanLike(Protocol):
    """Protocol describing a span-like object with integer start and end offsets."""

    start: int
    end: int


def span_start_end(span: Any) -> tuple[int, int]:
    """Extract integer start and end offsets from a span-like object."""
    if isinstance(span, tuple) and len(span) == 2:
        return int(span[0]), int(span[1])

    start = getattr(span, "start", None)
    end = getattr(span, "end", None)
    if isinstance(start, int) and isinstance(end, int):
        return start, end

    try:
        a, b = span
        return int(a), int(b)
    except Exception as exc:
        raise TypeError(f"Unrecognised span type: {type(span)!r} -> {span!r}") from exc


def build_definitions_by_acronym(
    *,
    extr: ExtractionResult,
    opts: ResolveOptions,
) -> dict[str, list[dict[str, Any]]]:
    """Build a mapping of acronym to ordered extracted definition candidates."""
    defs_by_ac: dict[str, list[dict[str, Any]]] = {}

    for key, pick in extr.picks.items():
        if pick is None:
            continue

        conf = float(pick.definition_confidence)
        if conf < float(opts.min_confidence):
            continue

        ds, de = span_start_end(pick.def_span)

        defs_by_ac.setdefault(key, []).append(
            {
                "text": pick.definition,
                "start": ds,
                "end": de,
                "confidence": conf,
                "source": "extracted",
                "_is_pick": True,
            }
        )

    for definition in extr.definitions:
        ac = definition.acronym
        conf = float(definition.definition_confidence)
        if conf < float(opts.min_confidence):
            continue

        cand = {
            "text": definition.definition,
            "start": int(definition.def_start),
            "end": int(definition.def_end),
            "confidence": conf,
            "source": "extracted",
            "_is_pick": False,
        }

        bucket = defs_by_ac.setdefault(ac, [])
        dedupe_key = (cand["text"], cand["start"], cand["end"])
        if not any((x["text"], x["start"], x["end"]) == dedupe_key for x in bucket):
            bucket.append(cand)

    max_k = int(opts.max_definitions_per_acronym)
    for ac, items in defs_by_ac.items():
        items.sort(key=lambda x: (not x["_is_pick"], -float(x["confidence"]), str(x["text"])))
        if max_k > 0:
            defs_by_ac[ac] = items[:max_k]
        for item in defs_by_ac[ac]:
            item.pop("_is_pick", None)

    return defs_by_ac


def maybe_glossary_block(
    *,
    glossary_repo: GlossaryRepository,
    acronym: str,
    lang: str,
    opts: ResolveOptions,
) -> dict[str, Any] | None:
    """Build the legacy glossary enrichment block for an acronym, if enabled."""
    if not opts.include_glossary_enrichment:
        return None

    meanings = glossary_repo.list_meanings(acronym=acronym)
    if not meanings:
        return None

    matches = [
        {
            "definition": str(m.get("definition") or ""),
            "domain": m.get("domain"),
            "lang": lang,
            "confidence": 1.0,
            "source": "system",
        }
        for m in meanings
        if bool(m.get("is_active")) and str(m.get("definition") or "").strip()
    ]

    if not matches:
        return None

    matches.sort(
        key=lambda x: (
            "" if x["domain"] is None else str(x["domain"]).casefold(),
            str(x["definition"]).casefold(),
        )
    )

    return {"matches": matches}


def map_pipeline_to_blocks(
    *,
    det_res: AcronymDetectorResult,
    extr: ExtractionResult,
    opts: ResolveOptions,
    lang: str,
    glossary_repo: GlossaryRepository,
) -> list[dict[str, Any]]:
    """Map detector and extraction outputs into public acronym response blocks."""
    occ_by_ac: dict[str, list[dict[str, int]]] = {}
    first_by_ac: dict[str, dict[str, int]] = {}

    for occurrence in det_res.occurrences:
        ac = occurrence.acronym
        occ_by_ac.setdefault(ac, []).append({"start": occurrence.start_offset, "end": occurrence.end_offset})

    for ac, occs in occ_by_ac.items():
        occs.sort(key=lambda s: (s["start"], s["end"]))
        first_by_ac[ac] = occs[0]

    defs_by_ac = build_definitions_by_acronym(extr=extr, opts=opts)

    acronyms_sorted = sorted(
        first_by_ac.keys(),
        key=lambda acronym: (first_by_ac[acronym]["start"], acronym),
    )

    blocks: list[dict[str, Any]] = []
    for ac in acronyms_sorted:
        first_occ = first_by_ac[ac]
        occs = occ_by_ac.get(ac, [])

        glossary_block = maybe_glossary_block(
            glossary_repo=glossary_repo,
            acronym=ac,
            lang=lang,
            opts=opts,
        )

        block: dict[str, Any] = {
            "acronym": ac,
            "first_occurrence": {"start": first_occ["start"], "end": first_occ["end"]},
            "definitions": defs_by_ac.get(ac, []),
        }

        if opts.return_occurrences:
            block["occurrences"] = occs

        if glossary_block is not None:
            block["glossary"] = glossary_block

        blocks.append(block)

    return blocks
