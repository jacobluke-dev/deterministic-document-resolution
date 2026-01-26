from __future__ import annotations

from dataclasses import dataclass

from plainera_unacronym.nlp.common.types import DetectorResult, DetectorConfig, Occurrence, FirstOccurrence
from plainera_unacronym.nlp.common.shared import normalize_acronym_key


@dataclass(frozen=True, slots=True)
class DroppedOccurrence:
    acronym: str
    start: int
    end: int
    rule: str
    detail: str


def post_detect_cleanup(
    text: str,
    det: DetectorResult,
    cfg: DetectorConfig,
) -> tuple[DetectorResult, str, list[DroppedOccurrence]]:
    """
    Post-detection cleanup (detect -> anchored boundary).

    Current rules (Tier-1, narrow & test-backed):
      - contained_suffix: drop an acronym that is strictly contained within another occurrence's span
        where the contained acronym is a strict suffix of the container acronym (case-insensitive).
        Ex: RNA (16,19) inside mRNA (15,19) -> drop RNA.
      - end_suffix_micro: if two different acronyms end at the same offset, and the shorter is a
        strict suffix of the longer, drop the shorter (even if starts differ enough to not be
        "contained" due to tokenisation oddities).
    """
    before = det.occurrences
    kept, dropped_a = _rule_inside_paren_suffix_of_left_acronym(text, before)
    kept, dropped_b = _rule_token_before_paren_suffix(text, kept)
    kept, dropped_c = _rule_contained_suffix(kept)
    kept, dropped_d = _rule_end_suffix_micro(kept)
    kept, dropped_e = _rule_drop_mixed_case_typos(kept)

    dropped = dropped_a + dropped_b + dropped_c + dropped_d + dropped_e

    # Recompute unique_acronyms from kept occurrences (authoritative boundary)
    firsts = _recompute_firsts(text, kept, cfg)

    cleaned = DetectorResult(unique_acronyms=firsts, occurrences=kept)
    summary = (
        f"cleanup occs {len(before)}→{len(kept)} "
        f"firsts={len(det.unique_acronyms)}→{len(firsts)} "
        f"dropped={len(dropped)}"
    )
    return cleaned, summary, dropped


def _is_alternating_case(acr: str) -> bool:
    # Consider only letters; digits/punct ignored
    letters = [c for c in acr if c.isalpha()]
    if len(letters) < 3:
        return False
    has_lower = any(c.islower() for c in letters)
    has_upper = any(c.isupper() for c in letters)
    if not (has_lower and has_upper):
        return False

    # Alternation if case flips on most adjacent transitions (>= 2 flips for len>=3)
    flips = 0
    for a, b in zip(letters, letters[1:]):
        if a.islower() != b.islower():
            flips += 1

    # For 3 letters: flips==2 => strict alternation (aBa)
    # For 4 letters: flips>=3 => aBaB
    return flips >= (len(letters) - 1)


def _rule_contained_suffix(occs: list[Occurrence]) -> tuple[list[Occurrence], list[DroppedOccurrence]]:
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

            # strictly contained by span
            if not (outer.start_offset <= inner.start_offset and inner.end_offset <= outer.end_offset):
                continue
            if outer.start_offset == inner.start_offset and outer.end_offset == inner.end_offset:
                continue  # exact dup not handled here

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

def _is_mixed_case_typo(acr: str) -> bool:
    letters = [c for c in acr if c.isalpha()]
    if len(letters) < 4:          # key: do NOT touch TfL (len 3) etc.
        return False

    upp = sum(c.isupper() for c in letters)
    low = sum(c.islower() for c in letters)
    if not (upp >= 3 and low == 1):
        return False

    if not letters[0].isupper():  # allow mRNA/iOS style
        return False

    # Find first lowercase after position 0
    first_low = None
    for i, c in enumerate(letters[1:], start=1):
        if c.islower():
            first_low = i
            break
    if first_low is None:
        return False

    # If there is an uppercase after that lowercase, it's an internal-case blip
    if any(c.isupper() for c in letters[first_low + 1:]):
        return True

    return False



def _rule_drop_mixed_case_typos(
    occs: list[Occurrence],
) -> tuple[list[Occurrence], list[DroppedOccurrence]]:
    ordered = sorted(occs, key=lambda o: (o.start_offset, o.end_offset, o.acronym))
    drop_ids: set[int] = set()
    dropped: list[DroppedOccurrence] = []

    for i, o in enumerate(ordered):
        if _is_mixed_case_typo(o.acronym):
            drop_ids.add(i)
            dropped.append(
                DroppedOccurrence(
                    acronym=o.acronym,
                    start=o.start_offset,
                    end=o.end_offset,
                    rule="drop_mixed_case_typo",
                    detail="mostly_upper_single_lower_or_alternating",
                )
            )

    kept = [o for i, o in enumerate(ordered) if i not in drop_ids]
    return kept, dropped




_PUNCT_TRIM = ".,;:)]}»”'\""

def _rule_inside_paren_suffix_of_left_acronym(text: str, occs: list[Occurrence]) -> tuple[list[Occurrence], list[DroppedOccurrence]]:
    ordered = sorted(occs, key=lambda o: (o.start_offset, o.end_offset, o.acronym))
    drop_ids: set[int] = set()
    dropped: list[DroppedOccurrence] = []

    # index by start offset for quick lookup
    by_start = {}
    for i, o in enumerate(ordered):
        by_start.setdefault(o.start_offset, []).append((i, o))

    for i, left in enumerate(ordered):
        # left token must be followed by '(' (allow ws)
        j = left.end_offset
        while j < len(text) and text[j].isspace():
            j += 1
        if j >= len(text) or text[j] != "(":
            continue

        # find occurrences that are fully inside the parens region after '('
        # quick-and-safe: only consider candidates whose start is after '(' and before the next ')'
        close = text.find(")", j + 1)
        if close == -1:
            continue

        for k, inner in enumerate(ordered):
            if k == i:
                continue
            if not (j + 1 <= inner.start_offset and inner.end_offset <= close):
                continue

            # candidate to drop: ALLCAPS alpha token inside parens
            inner_clean = inner.acronym.strip(_PUNCT_TRIM)
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


def _rule_end_suffix_micro(occs: list[Occurrence]) -> tuple[list[Occurrence], list[DroppedOccurrence]]:
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

        # Compare within group, prefer longer tokens
        group_sorted = sorted(group, key=lambda t: (-(t[1].end_offset - t[1].start_offset), t[1].start_offset, t[1].acronym))
        for a_idx, a in group_sorted:
            if a_idx in drop_idx:
                continue
            for b_idx, b in group_sorted:
                if b_idx == a_idx or b_idx in drop_idx:
                    continue
                # a is longer/equal first by sorting; drop b if b is strict suffix of a
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


def _is_strict_suffix(shorter: str, longer: str) -> bool:
    if len(shorter) >= len(longer):
        return False
    # Case-insensitive suffix match; keep it narrow (no fancy normalisation here)
    return longer.upper().endswith(shorter.upper())

def _rule_token_before_paren_suffix(
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

    # Index occurrences by start offset for quick "B starts right after '('"
    by_start: dict[int, list[Occurrence]] = {}
    for o in ordered:
        by_start.setdefault(o.start_offset, []).append(o)

    drop_ids: set[int] = set()
    dropped: list[DroppedOccurrence] = []

    # We'll use stable indices into `ordered` so we can drop deterministically.
    for idx, a in enumerate(ordered):
        if idx in drop_ids:
            continue

        # Only consider ALLCAPS-ish tokens as the "long-form tail word" noise.
        # Keep it narrow: "RNA", "HTTP", etc. (ignore mixed-case here)
        if not a.acronym.isupper():
            continue

        # Check for whitespace then '(' right after A
        j = a.end_offset
        ws = 0
        while j < len(text) and text[j].isspace() and ws < max_ws:
            j += 1
            ws += 1
        if j >= len(text) or text[j] != "(":
            continue

        # Find a candidate B occurrence that begins immediately after '(' (+ optional space)
        k = j + 1
        while k < len(text) and text[k].isspace():
            k += 1

        bs = by_start.get(k, [])
        if not bs:
            continue

        # Choose the "best" B: longest acronym at that start (more informative)
        b = max(bs, key=lambda o: (o.end_offset - o.start_offset, o.confidence))

        # Ensure B is inside the parentheses and matches suffix condition
        # (quick check for closing paren right after B, allowing whitespace)
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


def _recompute_firsts(
    text: str,
    occurrences: list[Occurrence],
    cfg: DetectorConfig,
) -> dict[str, FirstOccurrence]:
    firsts: dict[str, FirstOccurrence] = {}

    for o in occurrences:
        # Ensure we have a key
        k = o.normalized_key
        if not k:
            k = normalize_acronym_key(o.acronym, cfg.allow_chars, dotted_mode=cfg.dotted_display)
        if not k:
            continue

        prev = firsts.get(k)
        if prev is None or o.start_offset < prev.start_offset:
            firsts[k] = FirstOccurrence(
                acronym=o.acronym,
                start_offset=o.start_offset,
                end_offset=o.end_offset,
                confidence=o.confidence,
                normalized_key=k,
            )

    return firsts
