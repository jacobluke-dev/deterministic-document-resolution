from plainera_unacronym.nlp.common.types import Occurrence
from ..common import DroppedOccurrence


def _is_alternating_case(acr: str) -> bool:
    # Consider only letters; digits/punct ignored
    letters = [c for c in acr if c.isalpha()]
    if len(letters) < 3:
        return False

    has_lower = any(c.islower() for c in letters)
    has_upper = any(c.isupper() for c in letters)
    if not (has_lower and has_upper):
        return False

    flips = 0
    for a, b in zip(letters, letters[1:]):
        if a.islower() != b.islower():
            flips += 1

    # Strict alternation => flips == len-1 (e.g., aBa, aBaB)
    return flips == (len(letters) - 1)


def _is_mixed_case_typo(acr: str) -> bool:
    letters = [c for c in acr if c.isalpha()]
    if len(letters) < 4:          # key: do NOT touch TfL (len 3) etc.
        return False

    # Common OCR/typo: mostly uppercase with a single lowercase blip (not mRNA/iOS, etc.)
    upp = sum(c.isupper() for c in letters)
    low = sum(c.islower() for c in letters)
    if not (upp >= 3 and low == 1):
        # Also treat strict alternation as suspicious at len>=4 (e.g., AbCd)
        return _is_alternating_case(acr) and len(letters) >= 4

    if not letters[0].isupper():  # allow mRNA/iOS style
        return False

    first_low = None
    for i, c in enumerate(letters[1:], start=1):
        if c.islower():
            first_low = i
            break
    if first_low is None:
        return False

    # If there is an uppercase after that lowercase, it's an internal-case blip
    return any(c.isupper() for c in letters[first_low + 1:])


def rule_drop_mixed_case_typos(
    text: str,
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
