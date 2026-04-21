"""
Utilities for consuming left-hand numeric designators that correspond to
digit-prefixed acronyms (e.g., "fifth generation (5G)" or "5th generation (5G)").

Typical usage:
    tok_left = consume_left_numeric_designator(
        acr=acr,
        tokens=tokens,
        tok_left=tok_left,
        word_to_digits=WORD_TO_DIGITS,
    )
"""

import re
from collections.abc import Mapping, Sequence

# -----------------------------
# Common word→digit mappings
# -----------------------------

ORDINAL_WORDS: dict[str, str] = {
    "first": "1",
    "second": "2",
    "third": "3",
    "fourth": "4",
    "fifth": "5",
    "sixth": "6",
    "seventh": "7",
    "eighth": "8",
    "ninth": "9",
    "tenth": "10",
    # Useful extensions (standards/versions often go beyond 10)
    "eleventh": "11",
    "twelfth": "12",
    "thirteenth": "13",
    "fourteenth": "14",
    "fifteenth": "15",
    "sixteenth": "16",
    "seventeenth": "17",
    "eighteenth": "18",
    "nineteenth": "19",
    "twentieth": "20",
}

CARDINAL_WORDS: dict[str, str] = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}

WORD_TO_DIGITS: dict[str, str] = {
    **ORDINAL_WORDS,
    **CARDINAL_WORDS,
}

# -----------------------------
# Regexes
# -----------------------------

_LEADING_DIGITS_RE = re.compile(r"^(?P<n>\d+)")
_ORDINAL_NUM_RE = re.compile(r"^(?P<n>\d+)(?:st|nd|rd|th)$", re.IGNORECASE)
_EDGE_PUNCT_RE = re.compile(r"^[^\w]+|[^\w]+$")


def _matches_left_numeric_designator(
    prev_raw: str,
    want: str,
    *,
    word_to_digits: Mapping[str, str],
) -> bool:
    """Return whether a previous token matches the wanted leading digits."""
    prev = _EDGE_PUNCT_RE.sub("", prev_raw.strip().lower())
    if not prev:
        return False

    head = prev.split("-", 1)[0]

    m = _ORDINAL_NUM_RE.match(head)
    if m and m.group("n") == want:
        return True

    if head.isdigit() and head == want:
        return True

    return word_to_digits.get(head) == want


def consume_left_numeric_designator(
    *,
    acr: str,
    tokens: Sequence[str],
    tok_left: int,
    word_to_digits: Mapping[str, str] | None = None,
) -> int:
    """Optionally consume a left-hand numeric designator for digit-prefixed acronyms.

    Args:
        acr: Acronym surface form. Only digit-prefixed acronyms are eligible.
        tokens: Token stream for the candidate phrase.
        tok_left: Current left boundary token index (0-based).
        word_to_digits: Mapping of word numerics/ordinals to digits.

    Returns:
        ``tok_left - 1`` if the immediate left token matches the acronym's leading
        digits; otherwise ``tok_left``.
    """
    if word_to_digits is None:
        word_to_digits = WORD_TO_DIGITS
    if tok_left <= 0 or not acr:
        return tok_left

    m = _LEADING_DIGITS_RE.match(acr)
    if not m:
        return tok_left
    want = m.group("n")

    prev_raw = tokens[tok_left - 1]
    if not prev_raw.strip():
        return tok_left

    if prev_raw.rstrip().endswith(".") and tok_left < len(tokens):
        nxt = tokens[tok_left].lstrip()
        nxt0 = nxt[0] if nxt else ""
        if nxt0 in "\"'“”‘’([{" and len(nxt) > 1:
            nxt0 = nxt[1]
        if nxt0.isupper():
            return tok_left

    if _matches_left_numeric_designator(prev_raw, want, word_to_digits=word_to_digits):
        return tok_left - 1

    return tok_left
