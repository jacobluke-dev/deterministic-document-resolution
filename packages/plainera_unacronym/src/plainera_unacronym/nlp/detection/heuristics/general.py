from typing import TYPE_CHECKING, Collection, Union

from plainera_unacronym.nlp.common.constants import (
    BOUNDARY_TERMINATORS,
    CLOSING_QUOTES_BRACKETS,
    EXCLAMS,
    PLURAL_SUFFIXES_DEFAULT,
    POST_SPAN_TOKEN_RE,
)
from plainera_unacronym.nlp.common.types import DetectorConfig

if TYPE_CHECKING:
    from plainera_unacronym.nlp.heuristics.context import HeuristicCfg


def _alpha_len(s: str) -> int:
    return sum(ch.isalpha() for ch in s)


def strip_terminal_plural(surface: str) -> str:
    for suf in PLURAL_SUFFIXES_DEFAULT:
        if surface.endswith(suf) and surface[: -len(suf)].isupper():
            return surface[: -len(suf)]
    return surface


def is_all_caps_word(surface: str, allow_chars: Collection[str]) -> bool:
    """Check if a token is a single ALL-CAPS word (letters only, length ≥ 4).

    A valid ALL-CAPS word:
      * Contains at least 4 alphabetic characters.
      * Has no digits and no characters from `allow_chars` (e.g., '-', '&', '/').
      * All alphabetic characters are uppercase.

    Args:
        surface: The candidate token (e.g., "NASA", "RANDOM", "CPU").
        allow_chars: Characters that, if present in `surface`, disqualify it
            from being treated as a pure ALL-CAPS word.

    Returns:
        True if `surface` is an ALL-CAPS word under the above policy; otherwise False.
    """
    if _alpha_len(surface) < 4:
        return False
    if any(ch.isdigit() or ch in allow_chars for ch in surface):
        return False
    letters = [ch for ch in surface if ch.isalpha()]
    return bool(letters) and all(ch.isupper() for ch in letters)


def exclam_near_right(text: str, end: int, max_scan: int | None = None, stop_at_newline: bool = True) -> bool:
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
    i = s - 1
    # skip any whitespace
    while i >= 0 and text[i].isspace():
        i -= 1
    return i >= 0 and text[i] == ","


def _has_upper_after_with_fillers(text: str, start: int, max_fillers: int = 2) -> bool:
    fillers = 0
    for m in POST_SPAN_TOKEN_RE.finditer(text, start):
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
    surface: str, text: str, s: int, e: int, cfg: Union[DetectorConfig, "HeuristicCfg"]
) -> bool:
    if len(surface) < 4:
        return False
    if not is_all_caps_word(surface, cfg.allow_chars):
        return False
    if not (_comma_near_left(text, s) and exclam_near_right(text, e)):
        return False
    return _has_upper_after_with_fillers(text, e, max_fillers=2)


def is_in_caps_interjection_context_prev(
    surface: str, text: str, s: int, e: int, cfg: Union[DetectorConfig, "HeuristicCfg"]
) -> bool:
    """
    True if SURFACE is the second ALL-CAPS word (len≥3) in a shouty pair,
    and there is a preceding ALL-CAPS word (len≥4) after a nearby comma.
    Targets dropping 'THEN' in ', ALRIGHTY THEN!'.
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
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    seg = text[line_start:line_end].strip()
    letters = [c for c in seg if c.isalpha()]
    return len(letters) >= 6 and all(c.isupper() for c in letters)


def at_sentence_boundary(text: str, pos: int) -> bool:
    """True if `pos` looks like a sentence start (after terminators/closers/whitespace)."""
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
