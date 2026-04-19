from __future__ import annotations

from document_resolution.nlp.common.constants_regex import PUNCT_TRIM
from document_resolution.nlp.common.types import Occurrence

# TID252 circular imports
from ..post import DroppedOccurrence  # noqa: TID252


def _is_strict_suffix(shorter: str, longer: str) -> bool:
    """Return whether `shorter` is a strict case-insensitive suffix of `longer`.

    Args:
        shorter: Candidate suffix string.
        longer: Candidate superstring.

    Returns:
        True if `shorter` is shorter than `longer` and `longer` ends with it.
    """
    if len(shorter) >= len(longer):
        return False
    return longer.upper().endswith(shorter.upper())


def rule_contained_suffix(
    text: str,
    occs: list[Occurrence],
) -> tuple[list[Occurrence], list[DroppedOccurrence]]:
    """Drop occurrences contained within a longer occurrence when they are its strict suffix.

    Args:
        text: Source text, unused but kept for the rule-function contract.
        occs: Current occurrence list.

    Returns:
        Kept occurrences and drop records for removed suffix fragments.
    """
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
    """Drop shorter same-end occurrences when they are strict suffixes of longer ones.

    Args:
        text: Source text, unused but kept for the rule-function contract.
        occs: Current occurrence list.

    Returns:
        Kept occurrences and drop records for removed end-aligned suffix fragments.
    """
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
    """Drop ALLCAPS parenthetical acronyms that are strict suffixes of the left acronym.

    Args:
        text: Source text used to validate parentheses boundaries.
        occs: Current occurrence list.

    Returns:
        Kept occurrences and drop records for removed parenthetical suffix fragments.
    """

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


def _skip_ws(text: str, i: int, *, max_ws: int | None = None) -> int:
    """Advance `i` over whitespace, optionally capped by `max_ws`."""
    n = len(text)
    ws = 0
    while i < n and text[i].isspace() and (max_ws is None or ws < max_ws):
        i += 1
        ws += 1
    return i


def _best_occ_at_start(by_start: dict[int, list[Occurrence]], start: int) -> Occurrence | None:
    """Return the longest, then highest-confidence occurrence starting at `start`."""
    bs = by_start.get(start)
    if not bs:
        return None
    return max(bs, key=lambda o: (o.end_offset - o.start_offset, o.occurrence_confidence))


def _find_paren_occurrence_after(
    text: str,
    a: Occurrence,
    by_start: dict[int, list[Occurrence]],
    *,
    max_ws: int,
) -> Occurrence | None:
    """Find the occurrence immediately inside parentheses following `a`."""
    j = _skip_ws(text, a.end_offset, max_ws=max_ws)
    if j >= len(text) or text[j] != "(":
        return None

    k = _skip_ws(text, j + 1, max_ws=None)
    b = _best_occ_at_start(by_start, k)
    if not b:
        return None

    m = _skip_ws(text, b.end_offset, max_ws=None)
    if m >= len(text) or text[m] != ")":
        return None

    return b


def rule_token_before_paren_suffix(
    text: str,
    occs: list[Occurrence],
    *,
    max_ws: int = 2,
) -> tuple[list[Occurrence], list[DroppedOccurrence]]:
    """Drops an ALLCAPS token immediately before '(' when the parenthetical acronym ends with it.

    This rule removes “tail-word” fragment occurrences where an ALLCAPS token (A) directly
    precedes a parenthetical acronym (B), and B is a strict suffix-superstring match of A
    (case-insensitive). This commonly occurs when a long-form includes an ALLCAPS component
    right before introducing the acronym in parentheses.

    Example:
        "messenger RNA (mRNA)" -> drop "RNA" because "mRNA" endswith "RNA".

    Constraints (intentional narrowness):
        - A must be ALLCAPS (`a.acronym.isupper()`).
        - A must be followed by '(' allowing up to `max_ws` whitespace characters.
        - B must start immediately after '(' (allowing whitespace) and be a detected occurrence.
        - B must be immediately followed by ')' (allowing whitespace).
        - If multiple occurrences start at B's start offset, the longest/highest-confidence
          candidate is selected.

    Args:
        text: Source text used to validate parentheses boundaries and whitespace adjacency.
        occs: Current occurrence list from the cleanup pipeline.
        max_ws: Maximum number of whitespace characters allowed between A and '('.

    Returns:
        A tuple of:
          - kept: Occurrences with qualifying pre-parenthesis ALLCAPS tokens removed.
          - dropped: Drop records for each removed occurrence, with rule="token_before_paren_suffix".
    """
    ordered = sorted(occs, key=lambda o: (o.start_offset, o.end_offset, o.acronym))

    by_start: dict[int, list[Occurrence]] = {}
    for o in ordered:
        by_start.setdefault(o.start_offset, []).append(o)

    drop_ids: set[int] = set()
    dropped: list[DroppedOccurrence] = []

    for idx, a in enumerate(ordered):
        a_clean = a.acronym.strip(PUNCT_TRIM)
        letters = [c for c in a_clean if c.isalpha()]
        upp = sum(c.isupper() for c in letters)
        low = sum(c.islower() for c in letters)

        # Keep the rule narrow: drop only ALLCAPS or mixed-case tokens with >=2 uppers
        # (prevents dropping normal Title Case words like "Lamport").
        if idx in drop_ids:
            continue
        if len(letters) < 2:
            continue
        if not (a_clean.isupper() or (upp >= 2 and low >= 1)):
            continue

        b = _find_paren_occurrence_after(text, a, by_start, max_ws=max_ws)
        if not b:
            continue

        if not _is_strict_suffix(a.acronym, b.acronym):
            continue

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
