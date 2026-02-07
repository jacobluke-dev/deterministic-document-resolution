from plainera_unacronym.nlp.common.constants_regex import PUNCT_TRIM
from plainera_unacronym.nlp.common.types import Occurrence

from ..post import DroppedOccurrence


def _is_strict_suffix(shorter: str, longer: str) -> bool:
    """Returns True if `shorter` is a strict suffix of `longer` (case-insensitive).

    “Strict” means `shorter` must be strictly shorter than `longer`. Matching is
    performed using a simple case-insensitive `.endswith()` comparison; no other
    normalisation is applied (e.g., punctuation trimming is the caller’s responsibility).

    Args:
        shorter: Candidate suffix string.
        longer: Candidate superstring that may end with `shorter`.

    Returns:
        True if `longer` ends with `shorter` ignoring case and `len(shorter) < len(longer)`;
        otherwise False.

    Examples:
        _is_strict_suffix("RNA", "mRNA") -> True
        _is_strict_suffix("rna", "mRNA") -> True
        _is_strict_suffix("mRNA", "mRNA") -> False
        _is_strict_suffix("", "mRNA") -> True  # empty string is a strict suffix by definition
    """
    if len(shorter) >= len(longer):
        return False
    return longer.upper().endswith(shorter.upper())


def rule_contained_suffix(
    text: str,
    occs: list[Occurrence],
) -> tuple[list[Occurrence], list[DroppedOccurrence]]:
    """Drops occurrences strictly contained within a longer occurrence when they are a strict suffix.

    This rule removes fragment hits created by tokenisation/overlap where an acronym is
    fully contained inside a longer acronym span and is a strict suffix of the longer
    acronym (case-insensitive).

    Example:
        "mRNA" at (15, 19) and "RNA" at (16, 19) -> drop "RNA"
        because "RNA" is strictly contained and "mRNA".endswith("RNA") == True.

    Ordering:
        Input ordering is not assumed. Occurrences are sorted deterministically so that
        drop decisions and reports are stable across runs.

    Args:
        text: Source text (unused by this rule; included for the RuleFn contract).
        occs: Current occurrence list from the cleanup pipeline.

    Returns:
        A tuple of:
          - kept: Occurrences with contained strict-suffix fragments removed.
          - dropped: Drop records for each removed occurrence, with rule="contained_suffix".
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
    """Drops shorter occurrences that share an end offset and are strict suffixes of longer ones.

        This rule handles “micro-overlap” cases where two different occurrences end at the same
        character offset, but are not necessarily strictly contained due to tokenisation quirks.
        Within each `end_offset` group, if a shorter acronym is a strict suffix of a longer acronym
        (case-insensitive), the shorter occurrence is dropped.

        Example:
            Occurrences ending at 19:
              - "mRNA" (15, 19)
              - "RNA"  (16, 19)
            -> drop "RNA" because "mRNA" endswith "RNA".

        Ordering:
            - Input ordering is not assumed.
            - Occurrences are grouped by `end_offset`.
            - Within each group, longer occurrences are preferred as the “container” candidate.

        Args:
            text: Source text (unused by this rule; included for the RuleFn contract).
            occs: Current occurrence list from the cleanup pipeline.

        Returns:
            A tuple of:
              - kept: Occurrences with end-aligned strict-suffix fragments removed.
              - dropped: Drop records for each removed occurrence, with rule="end_suffix_micro".
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
    """Drops ALLCAPS acronyms inside parentheses when they are strict suffixes of the left acronym.

    This rule targets a common “alias clarification” pattern where an acronym is followed
    by a parenthetical containing a shorter ALLCAPS fragment that is merely a suffix of
    the left acronym.

    Example:
        "mRNA (RNA)" -> drop "RNA" because it is inside the parentheses and is a strict
        suffix of "mRNA".

    Constraints (intentional narrowness):
        - The left occurrence must be immediately followed by '(' allowing whitespace.
        - The inner occurrence must be fully inside the next ')' after that '('.
        - The inner token must be an ALLCAPS alphabetic acronym (len > 1) after
          punctuation trimming via `PUNCT_TRIM`.
        - Matching uses strict, case-insensitive suffix logic.

    Args:
        text: Source text used to locate parentheses boundaries.
        occs: Current occurrence list from the cleanup pipeline.

    Returns:
        A tuple of:
          - kept: Occurrences with qualifying parenthetical suffix fragments removed.
          - dropped: Drop records for each removed occurrence, with rule="inside_paren_suffix_of_left".
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
