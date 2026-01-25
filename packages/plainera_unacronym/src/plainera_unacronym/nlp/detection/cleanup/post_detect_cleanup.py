from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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

    kept, dropped_a = _rule_contained_suffix(before)
    kept, dropped_b = _rule_end_suffix_micro(kept)

    dropped = dropped_a + dropped_b

    # Recompute unique_acronyms from kept occurrences (authoritative boundary)
    firsts = _recompute_firsts(text, kept, cfg)

    cleaned = DetectorResult(unique_acronyms=firsts, occurrences=kept)
    summary = (
        f"cleanup occs {len(before)}→{len(kept)} "
        f"firsts={len(det.unique_acronyms)}→{len(firsts)} "
        f"dropped={len(dropped)}"
    )
    return cleaned, summary, dropped


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
