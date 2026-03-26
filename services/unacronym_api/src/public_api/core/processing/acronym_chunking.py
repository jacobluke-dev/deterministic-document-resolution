from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Chunk:
    start: int
    end: int
    text: str


def make_chunks(text: str, *, chunk_size: int, overlap: int) -> list[Chunk]:
    """
        Split `text` into overlapping windows suitable for chunked processing.

        Chunks are returned in order and use Python-slice semantics:
        each chunk covers the half-open interval [start, end), where `end` is exclusive.

        The next chunk starts at `previous_start + (chunk_size - overlap)`, ensuring an
        overlap region of `overlap` characters between consecutive chunks. Overlap is
        used to avoid missing matches (e.g. acronyms/definitions) that straddle chunk
        boundaries.

        Args:
            text: Full input text to chunk.
            chunk_size: Maximum number of characters per chunk. Must be > 0.
            overlap: Number of characters of overlap between consecutive chunks.
                Must satisfy 0 <= overlap < chunk_size.

        Returns:
            A list of `Chunk` objects in ascending order of `start`. For empty input,
            returns a single chunk with start=end=0.

        Raises:
            ValueError: If `chunk_size <= 0`, `overlap < 0`, or `overlap >= chunk_size`.

        Notes:
            - This function does not attempt to align chunks to word/sentence boundaries.
              It is purely character-based for determinism.
            - Coverage is complete: concatenating chunk ranges covers [0, len(text)]
              with possible overlaps but no gaps.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    n = len(text)
    if n == 0:
        return [Chunk(start=0, end=0, text="")]

    step = chunk_size - overlap
    chunks: list[Chunk] = []

    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(Chunk(start=start, end=end, text=text[start:end]))
        if end >= n:
            break
        start += step

    return chunks


def shift_blocks(blocks: list[dict[str, Any]], delta: int) -> list[dict[str, Any]]:
    """
    Shift all span offsets in mapped public blocks by a constant delta.

    This is used to convert per-chunk offsets (relative to the chunk text) back
    into global offsets (relative to the original full input text).

    The function shifts:
      - `first_occurrence.start/end`
      - each item in `occurrences` (if present)
      - each definition span in `definitions`

    Args:
        blocks: List of public blocks (AcronymBlock-shaped dicts) as produced by
            `_map_pipeline_to_blocks` for a chunk.
        delta: Integer offset to add to every start/end coordinate (typically the
            chunk's `start` index in the original text).

    Returns:
        A new list of blocks with shifted coordinates. If `delta == 0`, the input
        list is returned as-is for efficiency.

    Notes:
        - The function is intentionally tolerant: fields missing or of unexpected
          type are ignored rather than raising.
        - Offsets remain Python-slice semantics: `end` is exclusive after shifting.
    """
    if delta == 0:
        return blocks

    def _shift_span(span: dict[str, Any]) -> dict[str, Any]:
        return {"start": int(span["start"]) + delta, "end": int(span["end"]) + delta}

    out: list[dict[str, Any]] = []
    for b in blocks:
        nb = dict(b)

        if "first_occurrence" in nb and isinstance(nb["first_occurrence"], dict):
            nb["first_occurrence"] = _shift_span(nb["first_occurrence"])

        if "occurrences" in nb and isinstance(nb["occurrences"], list):
            nb["occurrences"] = [_shift_span(o) for o in nb["occurrences"] if isinstance(o, dict)]

        if "definitions" in nb and isinstance(nb["definitions"], list):
            defs = []
            for d in nb["definitions"]:
                if not isinstance(d, dict):
                    continue
                nd = dict(d)
                nd["start"] = int(nd["start"]) + delta
                nd["end"] = int(nd["end"]) + delta
                defs.append(nd)
            nb["definitions"] = defs

        out.append(nb)

    return out


def _block_acronym(b: dict[str, Any]) -> str | None:
    """
        Extract the acronym identifier from a block dict.

        Args:
            b: Candidate block dict.

        Returns:
            The acronym string if present and non-empty; otherwise None.

        Notes:
            - This helper centralises validation for grouping logic in `merge_blocks`.
    """
    ac = b.get("acronym")
    return ac if isinstance(ac, str) and ac else None


def _init_group_from_block(ac: str, b: dict[str, Any]) -> dict[str, Any]:
    """
        Initialise a merged-group record from a single acronym block.

        The returned dict is the mutable accumulator used during merge:
          - always includes `acronym`, `first_occurrence`, `definitions`
          - includes optional `occurrences` and `glossary` if present on the source block

        Args:
            ac: Acronym key for the group.
            b: Source block dict (AcronymBlock-shaped).

        Returns:
            A new dict suitable for accumulation/merging across chunks.

        Notes:
            - `first_occurrence` is copied defensively.
            - Definitions/occurrences are copied into new lists so the accumulator can
              be mutated without affecting inputs.
    """
    gb: dict[str, Any] = {
        "acronym": ac,
        "first_occurrence": dict(b.get("first_occurrence") or {}),
        "definitions": list(b.get("definitions") or []),
    }
    if "occurrences" in b:
        gb["occurrences"] = list(b.get("occurrences") or [])
    if "glossary" in b:
        gb["glossary"] = b["glossary"]
    return gb


def _merge_into_group(gb: dict[str, Any], b: dict[str, Any]) -> None:
    """
        Merge a single acronym block into an existing group accumulator.

        This appends:
          - `occurrences` (if present) into the group's occurrence list
          - `definitions` into the group's definition list

        It also copies `glossary` into the group if the group does not already have it.

        Args:
            gb: Group accumulator dict created by `_init_group_from_block`.
            b: Source block dict to merge in.

        Returns:
            None. Mutates `gb` in place.

        Notes:
            - Glossary enrichment is expected to be identical across chunks; this helper
              treats it as idempotent and retains the first seen.
    """
    if "occurrences" in b:
        gb.setdefault("occurrences", [])
        gb["occurrences"].extend(list(b.get("occurrences") or []))

    gb["definitions"].extend(list(b.get("definitions") or []))

    if "glossary" in b and "glossary" not in gb:
        gb["glossary"] = b["glossary"]


def _normalise_occurrences(nb: dict[str, Any]) -> None:
    """
        Deduplicate and sort occurrences on a merged block, updating first occurrence.

        Behaviour:
          - If `occurrences` exists and is a list, dedupe by `(start, end)` and sort by
            `(start, end)` ascending.
          - If any occurrences remain, set `first_occurrence` to the earliest occurrence.

        Args:
            nb: Block dict being normalised (mutated in place).

        Returns:
            None. Mutates `nb` in place.

        Notes:
            - This function assumes occurrences are Python-slice spans: end exclusive.
            - Invalid items (missing start/end) are ignored.
    """
    occ_list = nb.get("occurrences")
    if not isinstance(occ_list, list):
        return

    seen: set[tuple[int, int]] = set()
    occs: list[dict[str, int]] = []

    for o in occ_list:
        if not isinstance(o, dict) or "start" not in o or "end" not in o:
            continue
        key = (int(o["start"]), int(o["end"]))
        if key in seen:
            continue
        seen.add(key)
        occs.append({"start": key[0], "end": key[1]})

    occs.sort(key=lambda s: (s["start"], s["end"]))
    nb["occurrences"] = occs

    if occs:
        nb["first_occurrence"] = dict(occs[0])


def _normalise_definitions(nb: dict[str, Any]) -> list[dict[str, Any]]:
    """
        Deduplicate and sort definitions on a merged block deterministically.

        Behaviour:
          - Dedupe definitions by `(text, start, end)`.
          - Sort remaining definitions by:
              1) confidence descending (higher first)
              2) definition text ascending (tie-breaker)

        This mirrors the service-layer determinism rule used when mapping pipeline
        outputs (post-pick ordering), ensuring consistent output across chunk merges.

        Args:
            nb: Block dict being normalised (mutated in place).

        Returns:
            The normalised list of definition dicts (also assigned to `nb["definitions"]`).

        Notes:
            - Definitions that are not dicts are ignored.
            - If a definition is missing `confidence`, it is treated as 0.0.
    """
    raw = nb.get("definitions") or []
    seen: set[tuple[str, int, int]] = set()
    defs: list[dict[str, Any]] = []

    for d in raw:
        if not isinstance(d, dict):
            continue
        key = (str(d.get("text", "")), int(d.get("start", 0)), int(d.get("end", 0)))
        if key in seen:
            continue
        seen.add(key)
        defs.append(d)

    defs.sort(key=lambda x: (-float(x.get("confidence", 0.0)), str(x.get("text", ""))))
    nb["definitions"] = defs
    return defs


def _ensure_first_occurrence(nb: dict[str, Any], defs: list[dict[str, Any]]) -> bool:
    """
        Ensure `first_occurrence` is present and valid on a merged block.

        In chunked mode, `first_occurrence` is normally derived from occurrences (if
        returned). If occurrences are not returned, this helper provides a fallback.

        Behaviour:
          - If `first_occurrence` already contains integer `start` and `end`, do nothing.
          - Else, if `defs` is non-empty, derive `first_occurrence` from the earliest
            definition span by `(start, end)`.
          - Else, return False (block cannot be ordered deterministically).

        Args:
            nb: Block dict to validate/mutate.
            defs: Normalised definitions list for this block.

        Returns:
            True if `first_occurrence` is now valid; False if it cannot be derived.

        Notes:
            - The fallback is primarily defensive; the typical path is that occurrences
              exist and define `first_occurrence`.
    """
    fo = nb.get("first_occurrence") or {}
    ok = isinstance(fo, dict) and "start" in fo and "end" in fo
    if ok:
        return True

    if not defs:
        return False

    best = min(
        defs,
        key=lambda d: (int(d.get("start", 10**18)), int(d.get("end", 10**18))),
    )
    nb["first_occurrence"] = {"start": int(best["start"]), "end": int(best["end"])}
    return True


def merge_blocks(block_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """
        Merge multiple lists of acronym blocks (one list per chunk) deterministically.

        This function is used after chunked execution of the pipeline. Each chunk
        produces blocks with globalised offsets (via `shift_blocks`). Overlap between
        chunks will intentionally create duplicate detections; this merge step removes
        duplicates and produces a stable ordering.

        Merge rules:
          - Group blocks by `acronym`.
          - Occurrences (if present) are merged, deduped by `(start, end)`, and sorted.
            `first_occurrence` is set to the earliest occurrence.
          - Definitions are merged, deduped by `(text, start, end)`, and sorted by
            `(-confidence, text)` to match existing mapping determinism.
          - Glossary enrichment (if present) is copied once; it is treated as idempotent.
          - Final blocks are ordered by `(first_occurrence.start, acronym)`.

        Args:
            block_lists: A list of per-chunk block lists. Each inner list contains
                AcronymBlock-shaped dicts.

        Returns:
            A single list of merged blocks, deterministically ordered and deduplicated.

        Notes:
            - This function assumes blocks are already shifted into global coordinates.
            - Blocks without a valid `acronym` are ignored.
            - If a block lacks both a valid first occurrence and any definitions to
              derive one from, it is dropped to preserve determinism.
    """
    grouped: dict[str, dict[str, Any]] = {}

    for blocks in block_lists:
        for b in blocks:
            ac = _block_acronym(b)
            if ac is None:
                continue
            gb = grouped.get(ac)
            if gb is None:
                grouped[ac] = _init_group_from_block(ac, b)
            else:
                _merge_into_group(gb, b)

    merged: list[dict[str, Any]] = []
    for _, b in grouped.items():
        nb = dict(b)
        _normalise_occurrences(nb)
        defs = _normalise_definitions(nb)
        if not _ensure_first_occurrence(nb, defs):
            continue
        merged.append(nb)

    merged.sort(key=lambda b: (int(b["first_occurrence"]["start"]), str(b["acronym"])))
    return merged
