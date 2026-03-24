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
from collections.abc import Mapping
from typing import Sequence

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

# Optional: enable if you see "five generation (5G)" style in your corpora
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


def consume_left_numeric_designator(
    *,
    acr: str,
    tokens: Sequence[str],
    tok_left: int,
    word_to_digits: Mapping[str, str] = WORD_TO_DIGITS,
) -> int:
    """Optionally consume a left-hand numeric designator for digit-prefixed acronyms.

    If ``acr`` begins with one or more digits (e.g. ``"5G"``, ``"12V"``) and the
    token immediately to the left of ``tok_left`` expresses the same number,
    this function returns ``tok_left - 1`` to expand the candidate phrase window
    left by one token. Otherwise, it returns ``tok_left`` unchanged.

    Recognised forms for the left token (after light normalisation):
      - Numeric ordinals: ``"5th"``, ``"12th"`` (case-insensitive suffix)
      - Plain numerics: ``"5"``, ``"12"``
      - Word numerics/ordinals via ``word_to_digits``: ``"fifth" -> "5"``, ``"twelve" -> "12"``
      - Hyphenated tokens: if the token is hyphenated (e.g. ``"5th-generation"``),
        only the head segment (``"5th"``) is considered.

    Normalisation rules:
      - Strips *edge* punctuation (keeps internal hyphens).
      - Lowercases the left token for matching.

    Args:
        acr (str): Acronym surface form. Only digit-prefixed acronyms are eligible.
        tokens (Sequence[str]): Token stream for the candidate phrase.
        tok_left (int): Current left boundary token index (0-based).
        word_to_digits (Mapping[str, str]): Mapping of word numerics/ordinals to digits.

    Returns:
        int: ``tok_left - 1`` if the immediate left token matches the acronym's
        leading digits; otherwise ``tok_left``.

    Examples:
        >>> consume_left_numeric_designator(acr="5G", tokens=["fifth","generation"], tok_left=1)
        0
        >>> consume_left_numeric_designator(acr="12V", tokens=["12th","edition"], tok_left=1)
        0
        >>> consume_left_numeric_designator(acr="GPU", tokens=["graphics","processing"], tok_left=1)
        1
    """
    if tok_left <= 0 or not acr:
        return tok_left

    m = _LEADING_DIGITS_RE.match(acr)
    if not m:
        return tok_left

    want = m.group("n")  # full leading digit run (e.g., "12" in "12V")

    # Before normalisation
    prev_raw = tokens[tok_left - 1].strip().lower()
    if not prev_raw:
        return tok_left

    # If previous token ends with '.', only block consumption when the next token
    # looks like a new-sentence start (capitalised).
    if prev_raw.endswith(".") and tok_left < len(tokens):
        nxt = tokens[tok_left].lstrip()
        nxt0 = nxt[0] if nxt else ""
        if nxt0 in "\"'“”‘’([{" and len(nxt) > 1:
            nxt0 = nxt[1]
        if nxt0.isupper():
            return tok_left

    # normalisation: strip edge punctuation, keep internal hyphens
    prev = _EDGE_PUNCT_RE.sub("", prev_raw)

    # If tokeniser kept hyphens (e.g. "5th-generation"), look at head segment.
    head = prev.split("-", 1)[0]

    # Case 1: numeric ordinal token like "5th"
    m2 = _ORDINAL_NUM_RE.match(head)
    if m2 and m2.group("n") == want:
        return tok_left - 1

    # Case 2: plain numeric token like "5"
    if head.isdigit() and head == want:
        return tok_left - 1

    # Case 3: word mapping like "fifth" -> "5"
    if word_to_digits.get(head) == want:
        return tok_left - 1

    return tok_left
