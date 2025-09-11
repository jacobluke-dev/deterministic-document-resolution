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



def shouty_phrase_drop(surface: str, text: str, s: int, e: int, cfg: DetectorConfig) -> bool:
    if not is_all_caps_word(surface, cfg.allow_chars): return False
    if not (_comma_near_left(text, s) and exclam_near_right(text, e)): return False
    # drop only if part of a 2-word shouty phrase (ALRIGHTY THEN!)
    # naive: if next word is ALL-CAPS too
    i, n = e, len(text)
    while i < n and text[i].isspace(): i += 1
    j = i
    while j < n and text[j].isalpha(): j += 1
    nxt = text[i:j]
    return nxt.isupper() and len(nxt) >= 3


def is_all_caps_heading(text: str, start: int, end: int) -> bool:
    line_start = text.rfind("\n", 0, start) + 1
    line_end   = text.find("\n", end)
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
