import re
from typing import Iterator

from plainera_unacronym.nlp.config import (TRAILING_PUNCT,
                                           LEADING_BRACK,
                                           CLOSING_BRACK,
                                           STANDS_FOR_RE,
                                           APOSTROPHE_VARIANTS, BOUNDARY, TIME_RE)
from plainera_unacronym.nlp.types import DetectorConfig, pattern_cache


def compile_pattern(cfg: DetectorConfig) -> re.Pattern[str]:
    key = (cfg.min_len, cfg.max_len, cfg.allow_chars)
    if key in pattern_cache:
        return pattern_cache[key]

    sep = re.escape(cfg.allow_chars)  # allowed internal separators
    # Branch a: chunks separated by allowed punctuation, optional spaces around the sep.
    with_seps = rf"(?:[A-Z0-9]+(?:\s*[{sep}]\s*[A-Z0-9]+)+)"
    # Branch b: compact uppercase/alnum run with configurable length bounds.
    compact = rf"(?:[A-Z][A-Z0-9]{{{max(cfg.min_len - 1, 1)},{max(cfg.max_len - 1, 1)}}})"
    token = rf"(?P<tok>{with_seps}|{compact})"
    # Allow adjacency with brackets/quotes without consuming them.
    pattern = rf"{token}"

    compiled = re.compile(pattern)
    pattern_cache[key] = compiled
    return compiled


def _letters(token: str) -> str:
    return "".join(ch for ch in token if ch.isalpha())


def _caps_ratio(token: str) -> float:
    letters = _letters(token)
    if not letters:
        return 1.0
    upp = sum(1 for ch in letters if ch.isupper())
    return upp / len(letters)


def _strip_trailing_punct(text: str, start: int, end: int) -> tuple[int, int]:
    # Exclude common trailing punctuation from offsets.
    while end > start and text[end - 1] in TRAILING_PUNCT:
        end -= 1
    return start, end


def _in_brackets(text: str, start: int, end: int) -> tuple[bool, bool]:
    # (inside, adjacent)
    s = start
    e = end
    inside = (s > 0 and text[s - 1] in "([") and (e < len(text) and text[e] in ")]")
    adjacent = (s > 0 and text[s - 1] in LEADING_BRACK) or (e < len(text) and text[e] in CLOSING_BRACK)
    return inside, adjacent

# add
def has_stands_for_follow(text: str, end: int, max_chars: int = 24) -> bool:
    # Look only to the right, stop at sentence end or max_chars
    i, n = end, len(text)
    while i < n and text[i].isspace():
        i += 1
    j = i
    limit = min(n, i + max_chars)
    while j < limit and text[j] not in ".!?":
        j += 1
    return bool(STANDS_FOR_RE.search(text[i:j]))


def _has_stands_for_near(text: str, start: int, end: int, radius: int) -> bool:
    lo = max(0, start - radius)
    hi = min(len(text), end + radius)
    return bool(STANDS_FOR_RE.search(text[lo:hi]))


def _is_sentence_start(text: str, start: int) -> bool:
    # crude but fast: previous non-space is a sentence terminator or start of doc.
    i = start - 1
    while i >= 0 and text[i].isspace():
        i -= 1
    return i < 0 or text[i] in ".!?\n\r"


def _next_word_lowercase(text: str, end: int) -> bool:
    # Peek the immediate next word; if it's all-lowercase, return True.
    i = end
    n = len(text)
    while i < n and text[i].isspace():
        i += 1
    # skip leading quotes/brackets
    while i < n and text[i] in "\"'“”‘’([{":
        i += 1
    j = i
    while j < n and (text[j].isalpha() or text[j] == "'"):
        j += 1
    word = text[i:j]
    return bool(word) and word.islower()


def normalize_key(surface: str) -> str:
    # 1) normalize apostrophes
    s = "".join(APOSTROPHE_VARIANTS.get(ch, ch) for ch in surface)
    # 2) remove spaces *around* allowed separators only (R & D -> R&D), keep other spaces intact (rare).
    parts: list[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in "&/'’-":
            # swallow surrounding spaces
            if parts and parts[-1] == " ":
                parts.pop()
            parts.append(ch)
            i += 1
            while i < len(s) and s[i] == " ":
                i += 1
            continue
        parts.append(ch)
        i += 1
    return "".join(parts)


def core_len_for_bounds(token: str) -> int:
    # count alnum only for min/max length checks
    return sum(1 for ch in token if ch.isalnum())


def _at_sentence_boundary(text: str, pos: int) -> bool:
    i = pos - 1
    while i >= 0 and text[i].isspace():
        i -= 1
    return i < 0 or text[i] in BOUNDARY

def _prev_token(text: str, start: int) -> str:
    i = start - 1
    while i >= 0 and text[i].isspace():
        i -= 1
    j = i
    while j >= 0 and (text[j].isalnum() or text[j] in ":."):
        j -= 1
    return text[j+1:i+1]


def blacklist_context_drop(surface: str, text: str, start: int, end: int, cfg: DetectorConfig) -> bool:
    tok = surface

    # Only run if token is explicitly blacklisted OR is in the always-drop uppercase set
    if tok not in getattr(cfg, "blacklist", frozenset()) and tok not in cfg.non_acronym_upper:
        return False

    # Don’t drop in definition-y contexts
    inside, _ = _in_brackets(text, start, end)
    if inside or has_paren_definition(text, end) or has_stands_for_follow(text, end):
        return False

    if tok in cfg.non_acronym_upper:
        # “OK,” “OK.” or followed by lowercase word → drop (any position)
        i = end
        n = len(text)
        while i < n and text[i].isspace():
            i += 1
        if i < n and text[i] in ",.!?;:":
            return True
        if _next_word_lowercase(text, end):
            return True
        # fall through to generic fallback

    if tok == "IT":
        return _at_sentence_boundary(text, start) and _next_word_lowercase(text, end)

    if tok == "AM":
        prev = _prev_token(text, start)
        if TIME_RE.match(prev):  # keep shared constant if you have it
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

    # Generic fallback
    return _at_sentence_boundary(text, start) and _next_word_lowercase(text, end)



def has_paren_definition(text: str, end: int, max_chars: int = 80) -> bool:
    i, n = end, len(text)
    while i < n and text[i].isspace(): i += 1
    if i < n and text[i] == "(":
        j, alpha = i + 1, 0
        while j < n and (j - i) <= max_chars and text[j] != ")":
            if text[j].isalpha(): alpha += 1
            j += 1
        return j < n and alpha >= 5
    return False



def score(surface: str, text: str, start: int, end: int, cfg: DetectorConfig) -> float:
    score = 0.6
    inside, adjacent = _in_brackets(text, start, end)
    if inside: score += 0.25
    elif adjacent: score += 0.15
    if has_paren_definition(text, end): score += 0.25
    if has_stands_for_follow(text, end):
        score += 0.15
    if surface in cfg.soft_blacklist: score -= 0.2
    return max(0.0, min(1.0, score))


def context_window(text: str, start: int, end: int, window_chars: int) -> tuple[int, int]:
    # Prefer sentence boundaries; fall back to +/- window_chars.
    left = start
    while left > 0 and text[left - 1] not in ".!?\n\r":
        left -= 1
        if start - left >= window_chars:
            break

    right = end
    n = len(text)
    while right < n and text[right] not in ".!?\n\r":
        right += 1
        if right - end >= window_chars:
            break

    return left, right


def iter_candidates(text: str, cfg: DetectorConfig) -> Iterator[tuple[str, int, int]]:
    pat = compile_pattern(cfg)
    for m in pat.finditer(text):
        s, e = m.span("tok")
        s, e = _strip_trailing_punct(text, s, e)
        if e - s < cfg.min_len:  # quick guard
            continue
        surface = text[s:e]
        # bounds by core alnum length
        clen = core_len_for_bounds(surface)
        if clen < cfg.min_len or clen > cfg.max_len:
            continue
        # letter-case ratio
        if _caps_ratio(surface) < cfg.require_caps_ratio:
            continue
        # reject ultra-long cap runs even with punctuation
        if clen >= 15:
            continue
        # reject single-letter
        if clen == 1:
            continue
        yield surface, s, e


def iter_candidates_with(text: str, cfg: DetectorConfig, pat: re.Pattern[str]) -> Iterator[tuple[str, int, int]]:
    for m in pat.finditer(text):
        s, e = m.span("tok")
        s, e = _strip_trailing_punct(text, s, e)
        if e - s < cfg.min_len:
            continue
        surface = text[s:e]
        clen = core_len_for_bounds(surface)
        if clen < cfg.min_len or clen > cfg.max_len:
            continue
        if _caps_ratio(surface) < cfg.require_caps_ratio:
            continue
        if clen >= 15 or clen == 1:
            continue
        yield surface, s, e
