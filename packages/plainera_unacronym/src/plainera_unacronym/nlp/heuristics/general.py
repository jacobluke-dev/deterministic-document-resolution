from plainera_unacronym.nlp import DetectorConfig

from plainera_unacronym.nlp.config import BOUNDARY, TIME_RE, PLURAL_SUFFIXES, STANDS_FOR_RE, CLOSING_QUOTES_BRACKETS, \
    BOUNDARY_TERMINATORS, EXCLAMS
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
        print(i)
        print(text[i].isspace())
        print(text[i])
        i -= 1
    return i >= 0 and text[i] == ","


def is_in_caps_interjection_context(surface: str, text: str, s: int, e: int, cfg: DetectorConfig) -> bool:
    """
    True if SURFACE (ALL-CAPS word, len≥4) is followed by another ALL-CAPS word (len≥3)
    before an exclamation (e.g., ', ALRIGHTY THEN!').
    """
    if len(surface) < 4:  # avoid dropping CPU, GPU, etc.
        return False
    if not is_all_caps_word(surface, cfg.allow_chars):
        return False
    if not (_comma_near_left(text, s) and exclam_near_right(text, e)):
        return False

    # look ahead for next ALL-CAPS word (letters-only), allow up to 2 short fillers
    i, n = e, len(text)
    fillers = 0
    MAX_FILLERS = 2

    while i < n:
        ch = text[i]
        if ch == "!":
            return False
        if ch.isspace():
            i += 1;
            continue
        if ch.isalpha():
            j = i
            while j < n and text[j].isalpha():
                j += 1
            word = text[i:j]
            if not word.isupper():
                return False
            if len(word) >= 3:
                return True
            fillers += 1
            if fillers > MAX_FILLERS:
                return False
            i = j;
            continue
        # any other punctuation before '!' breaks the pattern
        return False
    return False


def is_in_caps_interjection_context_prev(surface: str, text: str, s: int, e: int, cfg: DetectorConfig) -> bool:
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
        prev_word = text[j + 1:i + 1]
        if prev_word.isupper() and len(prev_word) >= 4 and _comma_near_left(text, j + 1):
            return True
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

    # 0) Never drop inside/around definitions
    inside, _ = in_brackets(text, start, end)
    if inside or has_paren_definition(text, end) or has_stands_for_follow(text, end):
        return False

    # 1) Shouty ALL-CAPS interjection (“ALRIGHTY THEN!”)
    if (is_in_caps_interjection_context(surface, text, start, end, cfg)
        or is_in_caps_interjection_context_prev(surface, text, start, end, cfg)):
        return True

    # 1b) ALL-CAPS headings (skip)
    if is_all_caps_word(surface, cfg.allow_chars) and is_all_caps_heading(text, start, end):
        return True

    # 2) Token-specific polysemes (run BEFORE blacklist gates)
    if tok == "IT":
        return at_sentence_boundary(text, start) and next_word_lowercase(text, end)

    if tok == "AM":
        prev = prev_token(text, start)
        if TIME_RE.match(prev):  # e.g., "9 AM"
            return True
        # sentence-start "I AM …"
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

    # 3) From here on, only blacklisted / known non-acronym uppers
    if tok not in getattr(cfg, "blacklist", frozenset()) and tok not in cfg.non_acronym_upper:
        return False

    # 4) Known non-acronym uppers: drop on punctuation or lowercase continuation
    if tok in cfg.non_acronym_upper:
        i, n = end, len(text)
        while i < n and text[i].isspace():
            i += 1
        if i < n and text[i] in ",.!?;:":
            return True
        if next_word_lowercase(text, end):
            return True
        # fall through

    # 5) Generic fallback
    return at_sentence_boundary(text, start) and next_word_lowercase(text, end)
