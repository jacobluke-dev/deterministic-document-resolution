import re
from typing import Iterator

from plainera_unacronym.nlp.config import (TRAILING_PUNCT,
                                           LEADING_BRACK,
                                           CLOSING_BRACK,
                                           STANDS_FOR_RE,
                                           APOSTROPHE_VARIANTS, BOUNDARY, TIME_RE, PLURAL_SUFFIXES, DASH_MAP)
from plainera_unacronym.nlp.types import DetectorConfig, pattern_cache


def compile_pattern(cfg: DetectorConfig) -> re.Pattern[str]:
    key = (cfg.min_len, cfg.max_len, cfg.allow_chars, cfg.enable_dotted)
    if key in pattern_cache:
        return pattern_cache[key]

    sep = re.escape(cfg.allow_chars)

    # Branch a: chunks with allowed separators (R&D, USB-C, O’RAN, I/O)
    with_seps = rf"(?:[A-Z0-9]+(?:\s*[{sep}]\s*[A-Z0-9]+)+)"

    # Branch b: dotted initialisms (U.S., U.S.A.) — enabled via flag
    # - At least two dotted letters
    # - Optional trailing undotted letter
    # - Optional final dot
    dotted = r"(?:[A-Z]\.){2,}(?:[A-Z])?\.?"

    # Branch c: compact ALL-CAPS/alnum run (NHS, GPU, H2O)
    compact = rf"(?:[A-Z][A-Z0-9]{{{max(cfg.min_len-1, 1)},{max(cfg.max_len-1, 1)}}})"

    if cfg.enable_dotted:
        token = rf"(?P<tok>{with_seps}|{dotted}|{compact})"
    else:
        token = rf"(?P<tok>{with_seps}|{compact})"

    pat = re.compile(token)
    pattern_cache[key] = pat
    return pat


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



def strip_terminal_plural(surface: str) -> str:
    for suf in PLURAL_SUFFIXES:
        if surface.endswith(suf) and surface[:-len(suf)].isupper():
            return surface[:-len(suf)]
    return surface


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


def threshold_len(surface: str, allow_chars: str) -> int:
    """
    Effective length used for confidence thresholds.
    - Base is alnum length.
    - If the token contains any allowed internal separator (e.g. &, /, -),
      we treat it as at least length 3 so items like 'R&D' don't get penalised
      as two-letter tokens.
    """
    clen = core_len_for_bounds(surface)
    if any(ch in allow_chars for ch in surface) or "." in surface:
        return max(3, clen)
    return clen

def is_all_caps_heading(text: str, start: int, end: int) -> bool:
    line_start = text.rfind("\n", 0, start) + 1
    line_end   = text.find("\n", end)
    if line_end == -1: line_end = len(text)
    seg = text[line_start:line_end].strip()
    letters = [c for c in seg if c.isalpha()]
    return len(letters) >= 6 and all(c.isupper() for c in letters)



def normalize_key(surface: str, allow_chars: str, enable_dotted: bool = False) -> str:
    # 0) canonicalize look-alikes first
    s = "".join(APOSTROPHE_VARIANTS.get(ch, DASH_MAP.get(ch, ch)) for ch in surface)

    # 1) strip dots for dotted initialisms (safe: only dotted branch yields dots)
    if enable_dotted and "." in s:
        s = s.replace(".", "")

    # 2) swallow spaces around allowed internal separators (R & D -> R&D)
    parts: list[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in allow_chars:
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
    return text[j + 1:i + 1]


def blacklist_context_drop(surface: str, text: str, start: int, end: int, cfg: DetectorConfig) -> bool:
    tok = surface

    # 0) Don't drop if we're clearly in acronym-definition context
    inside, _ = _in_brackets(text, start, end)
    if inside or has_paren_definition(text, end) or has_stands_for_follow(text, end):
        return False

    # 1) Shouty ALL-CAPS phrase rule:
    #    two adjacent ALL-CAPS words, comma before the phrase, exclamation soon after.
    if _is_all_caps_word(surface, cfg.allow_chars) and _comma_near_left(text, start) and _exclam_near_right(text, end):
        if _next_all_caps_word(text, end, cfg.allow_chars) or _prev_all_caps_word(text, start, cfg.allow_chars):
            return True  # drops ALRIGHTY and THEN in "..., ALRIGHTY THEN!"

    # 1b) Sometimes titles can be all capitalised if so lets skip these!
    if _is_all_caps_word(surface, cfg.allow_chars) and is_all_caps_heading(text, start, end):
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
        if _next_word_lowercase(text, end):
            return True
        # fall through to generic

    # 4) Token-specific polysemes
    if tok == "IT":
        return _at_sentence_boundary(text, start) and _next_word_lowercase(text, end)

    if tok == "AM":
        prev = _prev_token(text, start)
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
    return _at_sentence_boundary(text, start) and _next_word_lowercase(text, end)


def _word_bounds_right(text: str, pos: int) -> tuple[int, int]:
    i, n = pos, len(text)
    while i < n and text[i].isspace(): i += 1
    j = i
    while j < n and (text[j].isalpha()): j += 1
    return i, j


def _word_bounds_left(text: str, pos: int) -> tuple[int, int]:
    i = pos - 1
    while i >= 0 and text[i].isspace(): i -= 1
    j = i
    while j >= 0 and (text[j].isalpha()): j -= 1
    return j + 1, i + 1


def _next_all_caps_word(text: str, end: int, allow_chars: str, max_gap: int = 2) -> bool:
    i, j = _word_bounds_right(text, end)
    return (0 <= i - end <= max_gap) and _is_all_caps_word(text[i:j], allow_chars)


def _prev_all_caps_word(text: str, start: int, allow_chars: str, max_gap: int = 2) -> bool:
    i, j = _word_bounds_left(text, start)
    return (0 <= start - j <= max_gap) and _is_all_caps_word(text[i:j], allow_chars)


def _alpha_len(s: str) -> int:
    return sum(ch.isalpha() for ch in s)


def _is_all_caps_word(surface: str, allow_chars: str) -> bool:
    # Single word, letters-only uppercase, no digits/separators
    if _alpha_len(surface) < 4:
        return False
    if any(ch.isdigit() or ch in allow_chars for ch in surface):
        return False
    letters = [ch for ch in surface if ch.isalpha()]
    return letters and all(ch.isupper() for ch in letters)


def _exclam_near_right(text: str, end: int, max_chars: int = 6) -> bool:
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


def has_letter(s: str) -> bool:
    return any(ch.isalpha() for ch in s)


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
    if inside:
        score += 0.25
    elif adjacent:
        score += 0.15
    if has_paren_definition(text, end): score += 0.25
    if has_stands_for_follow(text, end):
        score += 0.15
    if surface in cfg.soft_blacklist: score -= 0.2
    return max(0.0, min(1.0, score))


def context_window(text: str, start: int, end: int, window_chars: int) -> tuple[int, int]:
    # Left: back to previous terminator (or start), then skip spaces
    left = start
    while left > 0 and text[left - 1] not in ".!?\n\r":
        if start - left >= window_chars: break
        left -= 1
    while left < start and text[left].isspace():
        left += 1

    # Right: forward to next terminator (or end), include the terminator
    n = len(text)
    right = end
    while right < n and text[right] not in ".!?\n\r":
        if right - end >= window_chars: break
        right += 1
    if right < n and text[right] in ".!?\n\r":
        right += 1  # include the terminator

    return left, right


def iter_candidates(text: str, cfg: DetectorConfig) -> Iterator[tuple[str, int, int]]:
    pat = compile_pattern(cfg)
    for m in pat.finditer(text):
        s, e = m.span("tok")
        s, e = _strip_trailing_punct(text, s, e)
        if e - s < cfg.min_len:  # quick guard
            continue
        surface = text[s:e]
        if not has_letter(surface):
            continue
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
        if not has_letter(surface):
            continue
        clen = core_len_for_bounds(surface)
        if clen < cfg.min_len or clen > cfg.max_len:
            continue
        if _caps_ratio(surface) < cfg.require_caps_ratio:
            continue
        if clen >= 15 or clen == 1:
            continue
        yield surface, s, e


def reason_tags(surface: str, text: str, start: int, end: int, cfg: DetectorConfig) -> list[str]:
    tags: list[str] = []
    inside, adjacent = _in_brackets(text, start, end)
    if inside:   tags.append("inside_parens")
    elif adjacent: tags.append("adjacent_parens")
    if has_paren_definition(text, end):  tags.append("paren_definition_right")
    if has_stands_for_follow(text, end): tags.append("stands_for_right")
    if surface in cfg.soft_blacklist:    tags.append("soft_blacklist_penalty")
    if surface in cfg.non_acronym_upper: tags.append("non_acronym_upper")
    if _is_sentence_start(text, start):  tags.append("sentence_start")
    if _next_word_lowercase(text, end):  tags.append("next_word_lowercase")
    prev = _prev_token(text, start)
    if TIME_RE.match(prev):              tags.append("prev_time_token")
    if any(ch in cfg.allow_chars for ch in surface): tags.append("has_separator")
    if "." in surface and cfg.enable_dotted: tags.append("dotted_initialism")
    return tags
