from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Chunk:
    start: int
    end: int
    text: str


def make_chunks(text: str, *, chunk_size: int, overlap: int) -> list[Chunk]:
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
    """Shift start/end offsets in blocks by +delta (incl. first_occurrence)."""
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


def merge_blocks(block_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """
    Merge chunk blocks deterministically.

    - Group by acronym.
    - first_occurrence = min by (start,end) from occurrences if present else from first_occurrence.
    - occurrences: dedupe by (start,end), sort
    - definitions: dedupe by (text,start,end), then sort using SAME rule as service:
        (-confidence, text) with stable pick ordering already done per chunk.
      (We keep it consistent with current mapping rules, not “best of chunks”.)
    - blocks sorted by (first_occurrence.start, acronym)
    """
    grouped: dict[str, dict[str, Any]] = {}

    for blocks in block_lists:
        for b in blocks:
            ac = b.get("acronym")
            if not isinstance(ac, str) or not ac:
                continue

            gb = grouped.get(ac)
            if gb is None:
                gb = {
                    "acronym": ac,
                    "first_occurrence": dict(b.get("first_occurrence") or {}),
                    "definitions": list(b.get("definitions") or []),
                }
                if "occurrences" in b:
                    gb["occurrences"] = list(b.get("occurrences") or [])
                if "glossary" in b:
                    gb["glossary"] = b["glossary"]
                grouped[ac] = gb
                continue

            # merge occurrences
            if "occurrences" in b:
                gb.setdefault("occurrences", [])
                gb["occurrences"].extend(list(b.get("occurrences") or []))

            # merge definitions
            gb["definitions"].extend(list(b.get("definitions") or []))

            # merge glossary (idempotent; if present in any chunk it should be identical)
            if "glossary" in b and "glossary" not in gb:
                gb["glossary"] = b["glossary"]

    merged: list[dict[str, Any]] = []
    for ac, b in grouped.items():
        nb = dict(b)

        # --- occurrences dedupe + sort
        if "occurrences" in nb and isinstance(nb["occurrences"], list):
            seen_occ: set[tuple[int, int]] = set()
            occs = []
            for o in nb["occurrences"]:
                if not isinstance(o, dict) or "start" not in o or "end" not in o:
                    continue
                key = (int(o["start"]), int(o["end"]))
                if key in seen_occ:
                    continue
                seen_occ.add(key)
                occs.append({"start": key[0], "end": key[1]})
            occs.sort(key=lambda s: (s["start"], s["end"]))
            nb["occurrences"] = occs
            # first_occurrence should match the earliest occurrence
            if occs:
                nb["first_occurrence"] = dict(occs[0])

        # --- definitions dedupe + sort
        defs = []
        seen_def: set[tuple[str, int, int]] = set()
        for d in nb.get("definitions") or []:
            if not isinstance(d, dict):
                continue
            key = (str(d.get("text", "")), int(d.get("start", 0)), int(d.get("end", 0)))
            if key in seen_def:
                continue
            seen_def.add(key)
            defs.append(d)

        # Keep same ordering semantics as _build_definitions_by_acronym (post-pick):
        # (-confidence, text), stable across runs
        defs.sort(key=lambda x: (-float(x.get("confidence", 0.0)), str(x.get("text", ""))))
        nb["definitions"] = defs

        # Ensure first_occurrence exists even if occurrences aren’t returned
        fo = nb.get("first_occurrence") or {}
        if not (isinstance(fo, dict) and "start" in fo and "end" in fo):
            # fallback: derive from earliest definition span if present, else drop block
            if defs:
                best = min(defs, key=lambda d: (int(d.get("start", 10**18)), int(d.get("end", 10**18))))
                nb["first_occurrence"] = {"start": int(best["start"]), "end": int(best["end"])}
            else:
                continue

        merged.append(nb)

    merged.sort(key=lambda b: (int(b["first_occurrence"]["start"]), str(b["acronym"])))
    return merged
