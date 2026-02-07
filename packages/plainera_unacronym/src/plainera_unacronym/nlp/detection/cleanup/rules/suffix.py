# cleanup/rules/suffix.py
from __future__ import annotations

from plainera_unacronym.nlp.common.constants_regex import PUNCT_TRIM
from plainera_unacronym.nlp.common.types import Occurrence

from ..post import DroppedOccurrence


def _is_strict_suffix(shorter: str, longer: str) -> bool:
    if len(shorter) >= len(longer):
        return False
    return longer.upper().endswith(shorter.upper())


def rule_contained_suffix(
    text: str,
    occs: list[Occurrence],
) -> tuple[list[Occurrence], list[DroppedOccurrence]]:
    # Deterministic ordering helps stable drops/reports
    ordered = sorted(occs, key=lambda o: (o.start_offset, -(o.end_offset - o.start_offset), o.acronym))
    n = len(ordered)

    drop_idx: set[int] = set()
    dropped: list[DroppedOccurrence] = []

    for i in range(n):
        if i in drop_idx:
            continue
        outer = ordered[i]
        for j in range(n):
            if j == i or j in drop_idx:
                continue
            inner = ordered[j]
            if (outer.end_offset - outer.start_offset) <= (inner.end_offset - inner.start_offset):
                continue

            if not (outer.start_offset <= inner.start_offset and inner.end_offset <= outer.end_offset):
                continue
            if outer.start_offset == inner.start_offset and outer.end_offset == inner.end_offset:
                continue

            if _is_strict_suffix(inner.acronym, outer.acronym):
                drop_idx.add(j)
                dropped.append(
                    DroppedOccurrence(
                        acronym=inner.acronym,
                        start=inner.start_offset,
                        end=inner.end_offset,
                        rule="contained_suffix",
                        detail=f"contained_in={outer.acronym}@({outer.start_offset},{outer.end_offset})",
                    )
                )

    kept = [o for k, o in enumerate(ordered) if k not in drop_idx]
    return kept, dropped


def rule_end_suffix_micro(
    text: str,
    occs: list[Occurrence],
) -> tuple[list[Occurrence], list[DroppedOccurrence]]:
    ordered = sorted(occs, key=lambda o: (o.end_offset, o.start_offset, -(o.end_offset - o.start_offset), o.acronym))

    drop_idx: set[int] = set()
    dropped: list[DroppedOccurrence] = []

    # Group by end_offset; within group, if shorter is strict suffix of longer -> drop shorter
    i = 0
    while i < len(ordered):
        end = ordered[i].end_offset
        group: list[tuple[int, Occurrence]] = []
        while i < len(ordered) and ordered[i].end_offset == end:
            group.append((i, ordered[i]))
            i += 1

        # Prefer longer tokens
        group_sorted = sorted(
            group,
            key=lambda t: (-(t[1].end_offset - t[1].start_offset), t[1].start_offset, t[1].acronym),
        )
        for a_idx, a in group_sorted:
            if a_idx in drop_idx:
                continue
            for b_idx, b in group_sorted:
                if b_idx == a_idx or b_idx in drop_idx:
                    continue
                if _is_strict_suffix(b.acronym, a.acronym):
                    drop_idx.add(b_idx)
                    dropped.append(
                        DroppedOccurrence(
                            acronym=b.acronym,
                            start=b.start_offset,
                            end=b.end_offset,
                            rule="end_suffix_micro",
                            detail=f"suffix_of={a.acronym}@({a.start_offset},{a.end_offset})",
                        )
                    )

    kept = [o for k, o in enumerate(ordered) if k not in drop_idx]
    return kept, dropped


def rule_inside_paren_suffix_of_left_acronym(
    text: str,
    occs: list[Occurrence],
) -> tuple[list[Occurrence], list[DroppedOccurrence]]:
    ordered = sorted(occs, key=lambda o: (o.start_offset, o.end_offset, o.acronym))
    drop_ids: set[int] = set()
    dropped: list[DroppedOccurrence] = []

    for i, left in enumerate(ordered):
        # left token must be followed by '(' (allow ws)
        j = left.end_offset
        while j < len(text) and text[j].isspace():
            j += 1
        if j >= len(text) or text[j] != "(":
            continue

        close = text.find(")", j + 1)
        if close == -1:
            continue

        for k, inner in enumerate(ordered):
            if k == i:
                continue
            if not (j + 1 <= inner.start_offset and inner.end_offset <= close):
                continue

            inner_clean = inner.acronym.strip(PUNCT_TRIM)
            if not (inner_clean.isalpha() and inner_clean.isupper() and len(inner_clean) > 1):
                continue

            if _is_strict_suffix(inner_clean, left.acronym):
                drop_ids.add(k)
                dropped.append(
                    DroppedOccurrence(
                        acronym=inner.acronym,
                        start=inner.start_offset,
                        end=inner.end_offset,
                        rule="inside_paren_suffix_of_left",
                        detail=f"suffix_of={left.acronym}@({left.start_offset},{left.end_offset})",
                    )
                )

    kept = [o for idx, o in enumerate(ordered) if idx not in drop_ids]
    return kept, dropped


def rule_token_before_paren_suffix(
    text: str,
    occs: list[Occurrence],
    *,
    max_ws: int = 2,
) -> tuple[list[Occurrence], list[DroppedOccurrence]]:
    """
    Drop ALLCAPS token A immediately before '(' when an acronym B inside the parentheses
    is a strict suffix-superstring match (B endswith A, case-insensitive).

    Example: "messenger RNA (mRNA)" -> drop "RNA" because "mRNA" endswith "RNA".
    """
    ordered = sorted(occs, key=lambda o: (o.start_offset, o.end_offset, o.acronym))

    by_start: dict[int, list[Occurrence]] = {}
    for o in ordered:
        by_start.setdefault(o.start_offset, []).append(o)

    drop_ids: set[int] = set()
    dropped: list[DroppedOccurrence] = []

    for idx, a in enumerate(ordered):
        if idx in drop_ids:
            continue

        if not a.acronym.isupper():
            continue

        j = a.end_offset
        ws = 0
        while j < len(text) and text[j].isspace() and ws < max_ws:
            j += 1
            ws += 1
        if j >= len(text) or text[j] != "(":
            continue

        k = j + 1
        while k < len(text) and text[k].isspace():
            k += 1

        bs = by_start.get(k, [])
        if not bs:
            continue

        b = max(bs, key=lambda o: (o.end_offset - o.start_offset, o.confidence))

        m = b.end_offset
        while m < len(text) and text[m].isspace():
            m += 1
        if m >= len(text) or text[m] != ")":
            continue

        if _is_strict_suffix(a.acronym, b.acronym):
            drop_ids.add(idx)
            dropped.append(
                DroppedOccurrence(
                    acronym=a.acronym,
                    start=a.start_offset,
                    end=a.end_offset,
                    rule="token_before_paren_suffix",
                    detail=f"paren={b.acronym}@({b.start_offset},{b.end_offset})",
                )
            )

    kept = [o for i, o in enumerate(ordered) if i not in drop_ids]
    return kept, dropped
