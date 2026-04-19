from document_resolution.nlp.common.types import Occurrence

# TID252 circular imports
from ..common import DroppedOccurrence  # noqa: TID252


def _is_alternating_case(acr: str) -> bool:
    """Return whether the letters in `acr` strictly alternate case.

    Non-letters are ignored. Requires at least three letters and both uppercase
    and lowercase characters.

    Args:
        acr: Candidate acronym string.

    Returns:
        True if `acr` contains at least three letters and those letters strictly
        alternate between uppercase and lowercase; otherwise False.
    """
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
    """Return whether `acr` looks like a mixed-case typo or OCR artefact.

    Flags either mostly-uppercase forms with a single internal lowercase blip, or
    strictly alternating four-letter case patterns such as `AbCd`. Short mixed-case
    forms such as `TfL` are intentionally excluded.

    Non-letter characters are ignored for classification.

    Args:
        acr: Candidate acronym string.

    Returns:
        True if `acr` is likely a mixed-case typo/OCR artefact; otherwise False.
    """
    letters = [c for c in acr if c.isalpha()]
    if len(letters) < 4:  # key: do NOT touch TfL (len 3) etc.
        return False

    # Common OCR/typo: mostly uppercase with a single lowercase blip (not mRNA/iOS, etc.)
    upp = sum(c.isupper() for c in letters)
    low = sum(c.islower() for c in letters)
    if not (upp >= 3 and low == 1):
        # Strict alternation is a strong OCR/typo signal at 4 letters (e.g. "AbCd"),
        # but longer alternating stylised terms exist (e.g. "LaTeX"). Keep those.
        return _is_alternating_case(acr) and len(letters) == 4

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
    return any(c.isupper() for c in letters[first_low + 1 :])


def rule_drop_mixed_case_typos(
    text: str,
    occs: list[Occurrence],
) -> tuple[list[Occurrence], list[DroppedOccurrence]]:
    """Drop occurrences whose acronym shape looks like a mixed-case typo.

    Args:
        text: Source text, unused but kept for the rule-function contract.
        occs: Current occurrence list from the cleanup pipeline.

    Returns:
        Kept occurrences and drop records for removed items.
    """
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
