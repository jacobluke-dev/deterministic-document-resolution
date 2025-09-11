import re
from typing import Iterator

from plainera_unacronym.nlp import DetectorConfig
from plainera_unacronym.nlp.config import (TRAILING_PUNCT,
                                           LEADING_BRACK,
                                           CLOSING_BRACK,
                                           STANDS_FOR_RE,
                                           APOSTROPHE_VARIANTS,
                                           DASH_MAP,
                                           TIME_RE)
from plainera_unacronym.nlp.heuristics.shared import has_paren_definition
from plainera_unacronym.nlp.types import pattern_cache


def compile_pattern(cfg: DetectorConfig) -> re.Pattern[str]:
    """
    Build a linear, low-backtracking pattern that matches:
      1) Chunks with internal separators:    R&D, USB-C, O’RAN, I/O  (spaces around seps allowed)
      2) Dotted initialisms (opt-in):        U.S., U.S.A.
      3) Compact ALL-CAPS/alnum runs:        NHS, GPU, H2O
      4) CamelCaps (opt-in, upper-first):    TfL, eBPF  (requires ≥2 uppercase letters)

    We wrap the whole token in word boundaries (\b … \b) to avoid matching inside longer words.
    Note: \b is Unicode-aware in Python. Hyphens/quotes/dots are *inside* the token branches; the
    boundaries apply only to the token edges, so adjacency like NHS) or "NHS" still matches.
    """
    # Cache key must include all switches that change the pattern’s shape.
    # NOTE: if your config field is named `enable_mixed_case` (no underscore),
    key = (cfg.min_len, cfg.max_len, cfg.allow_chars, cfg.enable_dotted, cfg.enable_mixed_case)
    if key in pattern_cache:
        return pattern_cache[key]

    # Escape the set of allowed internal separators for the character class.
    sep = re.escape(cfg.allow_chars)

    # 1) Chunks with internal separators (R&D, USB-C, O’RAN, I/O).
    #    - Letters/digits on both sides of a separator from cfg.allow_chars.
    #    - Optional whitespace around the separator is allowed (e.g., "R & D").
    with_seps = rf"(?:[A-Z0-9]+(?:\s*[{sep}]\s*[A-Z0-9]+)+)"

    # 2) Dotted initialisms (opt-in).
    #    - One or more "LETTER + dot" pairs *followed by a final LETTER*.
    #      Ending on a letter keeps the right-hand \b boundary valid.
    #      Examples matched: "U.S", "U.S.A"  (the trailing period, if any, is
    #      left outside the match and later trimmed by strip_trailing_punct()).
    dotted = r"(?:[A-Z]\.)+[A-Z]"

    # 3) Compact ALL-CAPS/alnum runs within length bounds.
    #    - First char must be A–Z; remaining are A–Z or 0–9.
    #    - Length bounds derive from cfg.{min,max}_len (inclusive) and apply to
    #      the whole run (the first char counts toward the total).
    compact = rf"(?:[A-Z][A-Z0-9]{{{max(cfg.min_len-1,1)},{max(cfg.max_len-1,1)}}})"

    # 4) CamelCaps (opt-in, upper-first) for brand-style abbreviations.
    #    - Simple, linear pattern that captures tokens like "TfL", "eBPF" (upper-first only here).
    #    - We also guard this in the iterator by relaxing the caps ratio only if ≥2 uppers exist.
    camel_uc = r"(?:[A-Z][a-z]?){2,5}"

    # Order matters: keep more specific branches (with_seps/dotted) before the generic compact.
    branches = [with_seps, compact]
    if cfg.enable_dotted:
        branches.insert(1, dotted)    # give dotted precedence over compact
    if cfg.enable_mixed_case:
        branches.append(camel_uc)

    # Word boundaries prevent matching inside longer identifiers/words.
    # The branches themselves include internal punctuation; \b only applies at edges.
    token = r"\b(?P<tok>" + "|".join(branches) + r")\b"

    pat = re.compile(token)
    pattern_cache[key] = pat
    return pat


def letters(token: str) -> str:
    return "".join(ch for ch in token if ch.isalpha())


def caps_ratio(token: str) -> float:
    ls = letters(token)
    if not ls:
        return 1.0
    upp = sum(1 for ch in ls if ch.isupper())
    return upp / len(ls)


def strip_trailing_punct(text: str, start: int, end: int) -> tuple[int, int]:
    # Exclude common trailing punctuation from offsets.
    while end > start and text[end - 1] in TRAILING_PUNCT:
        end -= 1
    return start, end


def in_brackets(text: str, start: int, end: int) -> tuple[bool, bool]:
    # (inside, adjacent)
    s = start
    e = end
    inside = (s > 0 and text[s - 1] in "([") and (e < len(text) and text[e] in ")]")
    adjacent = (s > 0 and text[s - 1] in LEADING_BRACK) or (e < len(text) and text[e] in CLOSING_BRACK)
    return inside, adjacent


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

def is_sentence_start(text: str, start: int) -> bool:
    # crude but fast: previous non-space is a sentence terminator or start of doc.
    i = start - 1
    while i >= 0 and text[i].isspace():
        i -= 1
    return i < 0 or text[i] in ".!?\n\r"

def next_word_lowercase(text: str, end: int) -> bool:
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

def prev_token(text: str, start: int) -> str:
    i = start - 1
    while i >= 0 and text[i].isspace():
        i -= 1
    j = i
    while j >= 0 and (text[j].isalnum() or text[j] in ":."):
        j -= 1
    return text[j + 1:i + 1]


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


def has_letter(s: str) -> bool:
    return any(ch.isalpha() for ch in s)



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


def score(surface: str, text: str, start: int, end: int, cfg: DetectorConfig) -> float:
    score = 0.6
    inside, adjacent = in_brackets(text, start, end)
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


def _has_lower_and_upper(tok: str) -> bool:
    return any(c.islower() for c in tok if c.isalpha()) and any(c.isupper() for c in tok if c.isalpha())


def iter_candidates(text: str, cfg: DetectorConfig) -> Iterator[tuple[str,int,int]]:
    pat = compile_pattern(cfg)
    yield from iter_candidates_with(text, cfg, pat)

def iter_candidates_with(text: str, cfg: DetectorConfig, pat: re.Pattern[str]) -> Iterator[tuple[str,int,int]]:
    for m in pat.finditer(text):
        s, e = strip_trailing_punct(text, *m.span("tok"))
        if e - s < cfg.min_len: continue
        surface = text[s:e]
        if not has_letter(surface): continue
        clen = core_len_for_bounds(surface)
        if clen < cfg.min_len or clen > cfg.max_len: continue

        cap_ratio = caps_ratio(surface)
        req = cfg.require_caps_ratio
        if cfg.enable_mixed_case and _has_lower_and_upper(surface):
            upp = sum(1 for ch in surface if ch.isupper())
            if upp >= 2:  # avoid noise
                req = min(req, cfg.require_caps_ratio_mixed)
        if cap_ratio < req: continue
        if clen >= 15 or clen == 1: continue
        yield surface, s, e


def reason_tags(surface: str, text: str, start: int, end: int, cfg: DetectorConfig) -> list[str]:
    tags: list[str] = []
    inside, adjacent = in_brackets(text, start, end)
    if inside:   tags.append("inside_parens")
    elif adjacent: tags.append("adjacent_parens")
    if has_paren_definition(text, end):  tags.append("paren_definition_right")
    if has_stands_for_follow(text, end): tags.append("stands_for_right")
    if surface in cfg.soft_blacklist:    tags.append("soft_blacklist_penalty")
    if surface in cfg.non_acronym_upper: tags.append("non_acronym_upper")
    if is_sentence_start(text, start):  tags.append("sentence_start")
    if next_word_lowercase(text, end):  tags.append("next_word_lowercase")
    prev = prev_token(text, start)
    if TIME_RE.match(prev):              tags.append("prev_time_token")
    if any(ch in cfg.allow_chars for ch in surface): tags.append("has_separator")
    if "." in surface and cfg.enable_dotted: tags.append("dotted_initialism")
    return tags
