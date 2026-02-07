from plainera_unacronym.nlp.common.types import Occurrence
from ..common import DroppedOccurrence


def _is_alternating_case(acr: str) -> bool:
    """Returns True if the alphabetic characters in `acr` strictly alternate case.

    Non-letters (digits/punctuation) are ignored. The function requires at least three
    letters and the presence of both uppercase and lowercase characters. “Strict
    alternation” means every adjacent pair of letters flips case (e.g., aBa, aBaB).

    Examples:
        _is_alternating_case("aBa") -> True
        _is_alternating_case("AbCd") -> True
        _is_alternating_case("ABcD") -> False   # contains adjacent same-case letters
        _is_alternating_case("ABC") -> False    # no lowercase
        _is_alternating_case("a-bA") -> True    # '-' ignored

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
    """Heuristically flags mixed-case acronyms that look like internal-case typos/OCR artefacts.

    This predicate is intentionally conservative. It avoids touching short mixed-case
    acronyms such as "TfL" by requiring at least 4 alphabetic characters.

    A string is considered a likely typo if either:
      1) It is "mostly uppercase with a single lowercase blip" (>=3 uppercase letters and
         exactly 1 lowercase letter) *and* the lowercase letter is not the first letter
         (to allow "mRNA"/"iOS" style prefixes), and there is an uppercase letter after
         that lowercase (indicating an internal-case blip).
      2) It is strictly alternating case across letters at length >= 4 (e.g., "AbCd"),
         which is rarely a meaningful acronym shape and often indicates noise.

    Non-letter characters are ignored for classification.

    Args:
        acr: Candidate acronym string.

    Returns:
        True if `acr` is likely a mixed-case typo/OCR artefact; otherwise False.
    """
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
    """Drops occurrences whose acronym shape matches the mixed-case typo/OCR heuristic.

    This rule applies `_is_mixed_case_typo()` to each occurrence acronym and removes
    those considered likely internal-case artefacts (e.g., "ABCdE", "AbCd"). The rule
    is intentionally conservative and typically targets length >= 4 letter acronyms.

    Ordering:
        Input ordering is not assumed. Occurrences are sorted deterministically for
        stable keep/drop results and reporting.

    Args:
        text: Source text (unused by this rule; included for the RuleFn contract).
        occs: Current occurrence list from the cleanup pipeline.

    Returns:
        A tuple of:
          - kept: Occurrences with mixed-case typo candidates removed.
          - dropped: Drop records for each removed occurrence, with rule="drop_mixed_case_typo".
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
