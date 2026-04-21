"""
General-purpose detection heuristics.

These helpers provide small, fast context checks used during acronym detection (e.g. stripping
terminal plurals, identifying ALL-CAPS headings, sentence boundaries, and shouty interjection contexts).
They are designed to be pure and deterministic over `(surface, text, s, e, cfg)`.
"""

from collections.abc import Collection
from typing import TYPE_CHECKING, Union

from document_resolution.nlp.common.constants_regex import (
    BOUNDARY_TERMINATORS,
    CLOSING_QUOTES_BRACKETS,
    EXCLAMS,
    PLURAL_SUFFIXES_DEFAULT,
    POST_SPAN_TOKEN_ASCII_RE,
)
from document_resolution.nlp.common.types import AcronymDetectorConfig

if TYPE_CHECKING:
    from document_resolution.nlp.detection.heuristics.context import HeuristicCfg


def _alpha_len(s: str) -> int:
    """Count alphabetic characters in a string.

    Args:
        s: Input string.

    Returns:
        Number of characters where `ch.isalpha()` is True.
    """
    return sum(ch.isalpha() for ch in s)


def strip_terminal_plural(surface: str) -> str:
    """Strip a terminal plural/possessive suffix from ALL-CAPS acronyms.

    Only removes suffixes like "s", "'s", "’s" when the stem is fully uppercase and
    not a dotted initialism (e.g. "U.S.A.s" is left unchanged).

    Args:
        surface: Raw matched surface token.

    Returns:
        Surface with suffix removed, or original surface if no rule applies.
    """
    for suf in PLURAL_SUFFIXES_DEFAULT:
        if surface.endswith(suf):
            stem = surface[: -len(suf)]
            # Don't strip on dotted initialisms (U.S.A.s)
            if "." in stem:
                return surface
            if stem.isupper():
                return stem
    return surface


def is_all_caps_word(surface: str, allow_chars: Collection[str]) -> bool:
    """True if `surface` is a single ALL-CAPS “word” token (letters-only policy).

    Requires ≥4 alphabetic chars; digits or any `allow_chars` disqualify; all letters
    must be uppercase.

    Args:
        surface: Candidate token (e.g. "NASA").
        allow_chars: Characters that disqualify a token from being a pure caps word.

    Returns:
        True if the token matches the ALL-CAPS word policy, else False.
    """
    if _alpha_len(surface) < 4:
        return False
    if any(ch.isdigit() or ch in allow_chars for ch in surface):
        return False
    letters = [ch for ch in surface if ch.isalpha()]
    return bool(letters) and all(ch.isupper() for ch in letters)


def exclam_near_right(text: str, end: int, max_scan: int | None = None, stop_at_newline: bool = True) -> bool:
    """Check whether an exclamation appears to the right without a blocker.

    Looks for the nearest exclamation mark after `end`, returning False if a sentence
    blocker ('.', '?', and optionally newline) appears first.

    Args:
        text: Source text.
        end: Offset to start scanning from.
        max_scan: Optional max distance allowed to the exclamation.
        stop_at_newline: If True, a newline blocks the scan.

    Returns:
        True if an exclamation is found under the policy, else False.
    """
    rest = text[end:]
    # nearest exclamation
    bang_idxs = [rest.find(ch) for ch in EXCLAMS]
    bang_idx = min((i for i in bang_idxs if i != -1), default=None)
    if bang_idx is None:
        return False

    # any blocker before the exclamation?
    blockers = [rest.find("."), rest.find("?")]
    if stop_at_newline:
        blockers.append(rest.find("\n"))
    blocker_idx = min((i for i in blockers if i != -1), default=None)
    if blocker_idx is not None and blocker_idx < bang_idx:
        return False

    return max_scan is None or bang_idx <= max_scan


def _comma_near_left(text: str, s: int) -> bool:
    """True if a comma is the nearest non-space character immediately left of `s`.

    Args:
        text: Source text.
        s: Start offset of the candidate token.

    Returns:
        True if the previous non-whitespace char is ',', else False.
    """
    i = s - 1
    # skip any whitespace
    while i >= 0 and text[i].isspace():
        i -= 1
    return i >= 0 and text[i] == ","


def _has_upper_after_with_fillers(text: str, start: int, max_fillers: int = 2) -> bool:
    """Detect a shouty ALL-CAPS word soon after `start`, allowing small fillers.

    Scans tokens after `start`. Accepts patterns like "ALRIGHTY THEN!" by allowing up to
    `max_fillers` short ALL-CAPS words before a ≥3-letter ALL-CAPS word.

    Args:
        text: Source text.
        start: Offset to begin scanning from.
        max_fillers: Max number of short ALL-CAPS “filler” words allowed.

    Returns:
        True if a qualifying ALL-CAPS word appears before disqualifying punctuation, else False.
    """
    fillers = 0
    for m in POST_SPAN_TOKEN_ASCII_RE.finditer(text, start):
        tok = m.group(0)
        if tok.isspace():
            continue
        if tok == "!":
            return False
        if tok.isalpha():
            if not tok.isupper():
                return False
            if len(tok) >= 3:
                return True
            fillers += 1
            if fillers > max_fillers:
                return False
            continue
        # any other punctuation before '!' breaks the pattern
        return False
    return False


def is_in_caps_interjection_context(
    surface: str, text: str, s: int, e: int, cfg: Union[AcronymDetectorConfig, "HeuristicCfg"]
) -> bool:
    # Keep Union[...] here; `|` caused typing/runtime issues in this module.
    """
    Detect “shouty interjection” context like ', ALRIGHTY THEN!'.

    Requires `surface` to be an ALL-CAPS word and to sit between a left comma and a
    right exclamation, with an ALL-CAPS follow-up word.

    Args:
        surface: Candidate token.
        text: Source text.
        s: Start offset of `surface`.
        e: End offset of `surface`.
        cfg: Config providing `allow_chars`.

    Returns:
        True if `surface` matches the interjection context policy, else False.
    """
    if len(surface) < 4:
        return False
    if not is_all_caps_word(surface, cfg.allow_chars):
        return False
    if not (_comma_near_left(text, s) and exclam_near_right(text, e)):
        return False
    return _has_upper_after_with_fillers(text, e, max_fillers=2)


def is_in_caps_interjection_context_prev(
    surface: str, text: str, s: int, e: int, cfg: Union[AcronymDetectorConfig, "HeuristicCfg"]
) -> bool:
    # Keep Union[...] here; `|` caused typing/runtime issues in this module.
    """Detect the *second* ALL-CAPS word in a shouty pair (e.g. ', ALRIGHTY THEN!').

    Used to drop trailing words like 'THEN' when preceded by a nearby comma + ALL-CAPS
    word and followed by an exclamation.

    Args:
        surface: Candidate token (the second word).
        text: Source text.
        s: Start offset of `surface`.
        e: End offset of `surface`.
        cfg: Config providing `allow_chars`.

    Returns:
        True if `surface` matches the “second word in shouty pair” policy, else False.
    """
    if len(surface) < 3 or not is_all_caps_word(surface, cfg.allow_chars):
        return False
    if not exclam_near_right(text, e):
        return False

    # scan backwards for previous ALL-CAPS word, requiring a comma near the left of that word
    i = s - 1
    while i >= 0 and text[i].isspace():
        i -= 1
    # step over letters of previous word
    j = i
    while j >= 0 and text[j].isalpha():
        j -= 1
    if j < i:
        prev_word = text[j + 1 : i + 1]
        if prev_word.isupper() and len(prev_word) >= 4 and _comma_near_left(text, j + 1):
            return True
    return False


def is_all_caps_heading(text: str, start: int, end: int) -> bool:
    """
    True if the line containing `[start:end]` is an ALL-CAPS heading.

    Evaluates the full line; requires ≥6 letters and all letters uppercase.

    Args:
        text: Source text.
        start: Span start offset.
        end: Span end offset.

    Returns:
       True if the containing line is an ALL-CAPS heading, else False.
    """
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    seg = text[line_start:line_end].strip()
    letters = [c for c in seg if c.isalpha()]
    return len(letters) >= 6 and all(c.isupper() for c in letters)


def at_sentence_boundary(text: str, pos: int) -> bool:
    """Heuristic sentence-start check for `pos`.

    Scans left from `pos`, skipping whitespace/closers, then checks for a terminator
    cluster ('.', '!', '?', '…') or start-of-document.

    Args:
        text: Source text.
        pos: Candidate sentence-start position.

    Returns:
        True if `pos` appears to be at a sentence boundary, else False.
    """
    i = pos - 1

    # Swallow any mix of whitespace and closing quotes/brackets (in any order).
    while i >= 0 and (text[i].isspace() or text[i] in CLOSING_QUOTES_BRACKETS):
        i -= 1

    # Allow clusters like "?!", "…", "!!!"
    saw_term = False
    while i >= 0 and text[i] in BOUNDARY_TERMINATORS:
        saw_term = True
        i -= 1

    # Start-of-doc also counts.
    return saw_term or i < 0
