from plainera_unacronym.nlp import DetectorConfig

from plainera_unacronym.nlp.config import BOUNDARY, TIME_RE, PLURAL_SUFFIXES, STANDS_FOR_RE, CLOSING_QUOTES_BRACKETS, \
    BOUNDARY_TERMINATORS
from plainera_unacronym.nlp.heuristics.core import in_brackets, has_stands_for_follow, next_word_lowercase, prev_token
from plainera_unacronym.nlp.heuristics.shared import has_paren_definition


def _alpha_len(s: str) -> int:
    return sum(ch.isalpha() for ch in s)


def has_stands_for_near(text: str, start: int, end: int, radius: int) -> bool:
    lo = max(0, start - radius)
    hi = min(len(text), end + radius)
    return bool(STANDS_FOR_RE.search(text[lo:hi]))


def strip_terminal_plural(surface: str) -> str:
    for suf in PLURAL_SUFFIXES:
        if surface.endswith(suf) and surface[:-len(suf)].isupper():
            return surface[:-len(suf)]
    return surface


def is_all_caps_word(surface: str, allow_chars: str) -> bool:
    # Single word, letters-only uppercase, no digits/separators
    if _alpha_len(surface) < 4:
        return False
    if any(ch.isdigit() or ch in allow_chars for ch in surface):
        return False
    letters = [ch for ch in surface if ch.isalpha()]
    return letters and all(ch.isupper() for ch in letters)


def exclam_near_right(text: str, end: int, max_chars: int = 6) -> bool:
    i, n = end, len(text)
    while i < n and i - end <= max_chars:
        if text[i] == "!":
            return True
        if text[i] in ".?":
            return False
        i += 1
    return False


def _comma_near_left(text: str, start: int, max_chars: int = 8) -> bool:
    i = start - 1
    steps = 0
    while i >= 0 and steps <= max_chars:
        if text[i] == ",":
            return True
        if text[i] in ".!?":
            return False
        if not text[i].isspace():
            steps += 1
        i -= 1
    return False


def word_bounds_right(text: str, pos: int) -> tuple[int, int]:
    i, n = pos, len(text)
    while i < n and text[i].isspace(): i += 1
    j = i
    while j < n and (text[j].isalpha()): j += 1
    return i, j


def word_bounds_left(text: str, pos: int) -> tuple[int, int]:
    i = pos - 1
    while i >= 0 and text[i].isspace(): i -= 1
    j = i
    while j >= 0 and (text[j].isalpha()): j -= 1
    return j + 1, i + 1


def _prev_all_caps_word(text: str, start: int, allow_chars: str, max_gap: int = 2) -> bool:
    i, j = word_bounds_left(text, start)
    return (0 <= start - j <= max_gap) and is_all_caps_word(text[i:j], allow_chars)


def next_all_caps_word(text: str, end: int, allow_chars: str, max_gap: int = 2) -> bool:
    i, j = word_bounds_right(text, end)
    return (0 <= i - end <= max_gap) and is_all_caps_word(text[i:j], allow_chars)


def is_in_caps_interjection_context(surface: str, text: str, s: int, e: int, cfg: DetectorConfig) -> bool:
    """
    Return True when an ALL-CAPS token appears to be part of a two-word “shouty
    phrase” like “ALRIGHTY THEN!” so the token can be safely dropped from
    acronym detection.

    The pattern this targets is:

        <comma near the left>  SURFACE  <spaces>  NEXT  <exclamation near right>

    where:
      * SURFACE is an ALL-CAPS “word” per `is_all_caps_word(surface, cfg.allow_chars)`.
      * NEXT is the next sequence of alphabetic characters (Unicode-aware) after
        the span [s:e], must be ALL-CAPS and length ≥ 3.
      * “near” is decided by helper predicates `_comma_near_left(text, s)` and
        `exclam_near_right(text, e)` (they typically enforce small max gaps).

    This is a cheap, punctuation-driven heuristic for ignoring emphatic interjections
    (e.g., “Well, ALRIGHTY THEN!”) that are unlikely to be acronyms.

    Args:
      surface: The text covered by the candidate span `[s:e]`; usually `text[s:e]`.
      text: Full source string that contains the candidate span.
      s: Start index (inclusive) of the candidate token within `text`.
      e: End index (exclusive) of the candidate token within `text`.
      cfg: Detector configuration. `cfg.allow_chars` is forwarded to
        `is_all_caps_word` to allow internal separators (e.g., `R&D`, `GPU/CPU`,
        `MOVE-ON`, `A_B`, `A.B`). Depending on your implementation of
        `is_all_caps_word`, other config fields (e.g., soft blacklists like
        common function words: “OF”, “IN”, “GO”) may cause `surface` to be
        rejected even if it is uppercase.

    Returns:
      bool: True if the token at `[s:e]` should be dropped because it matches
      the two-word shouty pattern; False otherwise.

    Notes:
      * The next word is parsed as consecutive `str.isalpha()` characters starting
        at the first non-space after `e`. Digits or hyphens in the *next* word are
        not considered here.
      * Multiple exclamation marks are okay (e.g., “THEN!!”), as long as
        `exclam_near_right` considers them “near”.
      * Whitespace between `SURFACE` and `NEXT` is ignored.
      * This heuristic is independent of sentence boundaries; it keys only off
        local punctuation.

    Examples:
      >>> text = "Well, ALRIGHTY THEN!"
      >>> s, e = text.index("ALRIGHTY"), text.index("ALRIGHTY") + len("ALRIGHTY")
      >>> is_in_caps_interjection_context("ALRIGHTY", text, s, e, cfg)  # doctest: +SKIP
      True

      >>> text = "Well ALRIGHTY THEN!"  # no comma near left
      >>> s, e = text.index("ALRIGHTY"), text.index("ALRIGHTY") + len("ALRIGHTY")
      >>> is_in_caps_interjection_context("ALRIGHTY", text, s, e, cfg)  # doctest: +SKIP
      False

      >>> text = "Well, ALRIGHTY Then!"  # next word not ALL-CAPS
      >>> s, e = text.index("ALRIGHTY"), text.index("ALRIGHTY") + len("ALRIGHTY")
      >>> is_in_caps_interjection_context("ALRIGHTY", text, s, e, cfg)  # doctest: +SKIP
      False

      >>> text = "Well, MOVE NOW!"  # passes if MOVE is allowed by your config
      >>> s, e = text.index("MOVE"), text.index("MOVE") + len("MOVE")
      >>> is_in_caps_interjection_context("MOVE", text, s, e, cfg)  # doctest: +SKIP
      True
    """
    if not is_all_caps_word(surface, cfg.allow_chars):
        return False
    if not (_comma_near_left(text, s) and exclam_near_right(text, e)):
        return False

    i, n = e, len(text)
    fillers = 0
    MAX_FILLERS = 2  # allow e.g. "I", "AM" before the content word

    while i < n:
        # stop early if we reach the exclamation
        if text[i] == "!":
            return False
        # skip spaces
        while i < n and text[i].isspace():
            i += 1
        # read the next alphabetic word
        j = i
        while j < n and text[j].isalpha():
            j += 1
        nxt = text[i:j]
        if not nxt:
            # non-letter (punctuation) — treat as not a valid shouty phrase
            return False

        if nxt.isupper():
            if len(nxt) >= 3:
                return True  # e.g. "HELLO I AM COOL!"
            # short ALL-CAPS filler like "I", "AM"
            fillers += 1
            if fillers > MAX_FILLERS:
                return False
            i = j
            continue
        # mixed/lowercase breaks the shouty run
        return False

    return False



def is_all_caps_heading(text: str, start: int, end: int) -> bool:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1: line_end = len(text)
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


def blacklist_context_drop(surface: str, text: str, start: int, end: int, cfg: DetectorConfig) -> bool:
    tok = surface

    # 0) Don't drop if we're clearly in acronym-definition context
    inside, _ = in_brackets(text, start, end)
    if inside or has_paren_definition(text, end) or has_stands_for_follow(text, end):
        return False

    # 1) Shouty ALL-CAPS phrase rule:
    #    two adjacent ALL-CAPS words, comma before the phrase, exclamation soon after.
    if is_all_caps_word(surface, cfg.allow_chars) and _comma_near_left(text, start) and exclam_near_right(text, end):
        if next_all_caps_word(text, end, cfg.allow_chars) or _prev_all_caps_word(text, start, cfg.allow_chars):
            return True  # drops ALRIGHTY and THEN in "..., ALRIGHTY THEN!"

    # 1b) Sometimes titles can be all capitalised if so lets skip these!
    if is_all_caps_word(surface, cfg.allow_chars) and is_all_caps_heading(text, start, end):
        return True

    # 2) From here on, only run for blacklisted / known non-acronym uppers
    if tok not in getattr(cfg, "blacklist", frozenset()) and tok not in cfg.non_acronym_upper:
        return False

    # 3) Non-acronym uppercase (e.g., OK, LTD, PLC) drop unless punctuation/lowercase context says otherwise
    if tok in cfg.non_acronym_upper:
        i, n = end, len(text)
        while i < n and text[i].isspace(): i += 1
        if i < n and text[i] in ",.!?;:":  # "OK," / "OK." etc.
            return True
        if next_word_lowercase(text, end):
            return True
        # fall through to generic

    # 4) Token-specific polysemes
    if tok == "IT":
        return at_sentence_boundary(text, start) and next_word_lowercase(text, end)

    if tok == "AM":
        prev = prev_token(text, start)
        if TIME_RE.match(prev):  # time-of-day
            return True
        # “I AM …” with boundary before I
        i = start - 1
        while i >= 0 and text[i].isspace():
            i -= 1
        if i >= 0 and text[i] == "I":
            j = i - 1
            while j >= 0 and text[j].isspace():
                j -= 1
            if j < 0 or text[j] in BOUNDARY:
                return True
        return False

    # 5) Generic fallback: sentence-start + next word lowercase
    return at_sentence_boundary(text, start) and next_word_lowercase(text, end)
