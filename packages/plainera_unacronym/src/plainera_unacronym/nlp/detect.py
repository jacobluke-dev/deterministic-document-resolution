
import dataclasses
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Tuple, Dict

# ---------------------------
# Configuration
# ---------------------------

@dataclass(frozen=True)
class DetectorConfig:
    min_len: int = 2
    max_len: int = 10
    # Allowed internal punctuation in acronyms (normalized for keying).
    allow_chars: str = "&/'’-"
    # Very small, locale-aware blacklist. Configurable/overrideable.
    blacklist: frozenset[str] = frozenset({"AM", "OK", "NO", "IT"})
    locale: str = "en_GB"
    window_chars: int = 80
    # Letters only: ratio of uppercase letters over letters (digits ignored).
    require_caps_ratio: float = 0.7


@dataclass(frozen=True)
class Occurrence:
    acronym: str                 # surface form as detected (not lowercased)
    start_offset: int
    end_offset: int              # end-exclusive
    confidence: float
    context_window: Tuple[int, int]


@dataclass(frozen=True)
class FirstOccurrence:
    acronym: str
    start_offset: int
    end_offset: int
    confidence: float


@dataclass(frozen=True)
class DetectorResult:
    unique_acronyms: Dict[str, FirstOccurrence]   # key = normalized_key
    occurrences: List[Occurrence]


# ---------------------------
# Regex compilation
# ---------------------------

# Notes:
# 1) Two branches:
#    a) Chunk-with-separators: R & D, O’RAN, R-D, R/D  (spaces around separators allowed)
#    b) Compact ALL-CAPS/alnum: NHS, GPU, H2O, G8, MP3
#
# 2) We purposely do *not* include trailing punctuation (,.;:!?) in the match.
#    We'll slice it away if present, conservatively.
#
# 3) We avoid catastrophic backtracking by keeping the pattern simple and linear.

# Will be rebuilt on first call per-config, but cached at module level per (pattern_str -> Pattern).
_pattern_cache: Dict[Tuple[int, int, str], re.Pattern[str]] = {}

def _compile_pattern(cfg: DetectorConfig) -> re.Pattern[str]:
    key = (cfg.min_len, cfg.max_len, cfg.allow_chars)
    if key in _pattern_cache:
        return _pattern_cache[key]

    sep = re.escape(cfg.allow_chars)  # allowed internal separators
    # Branch a: chunks separated by allowed punctuation, optional spaces around the sep.
    with_seps = rf"(?:[A-Z0-9]+(?:\s*[{sep}]\s*[A-Z0-9]+)+)"
    # Branch b: compact uppercase/alnum run with configurable length bounds.
    compact  = rf"(?:[A-Z][A-Z0-9]{{{max(cfg.min_len-1, 1)},{max(cfg.max_len-1, 1)}}})"
    token    = rf"(?P<tok>{with_seps}|{compact})"
    # Allow adjacency with brackets/quotes without consuming them.
    pattern  = rf"{token}"

    compiled = re.compile(pattern)
    _pattern_cache[key] = compiled
    return compiled


# ---------------------------
# Helpers
# ---------------------------

_APOSTROPHE_VARIANTS = {"'": "’", "’": "’"}  # normalize to curly for keying

_TRAILING_PUNCT = ",.;:!?)]}»”"
_LEADING_BRACK  = "([«“["
_CLOSING_BRACK  = ")]»”]"

_STANDS_FOR_RE  = re.compile(r"\bstands\s+for\b", re.IGNORECASE)

def _letters(token: str) -> str:
    return "".join(ch for ch in token if ch.isalpha())

def _caps_ratio(token: str) -> float:
    letters = _letters(token)
    if not letters:
        return 1.0
    upp = sum(1 for ch in letters if ch.isupper())
    return upp / len(letters)

def _strip_trailing_punct(text: str, start: int, end: int) -> Tuple[int, int]:
    # Exclude common trailing punctuation from offsets.
    while end > start and text[end - 1] in _TRAILING_PUNCT:
        end -= 1
    return start, end

def _in_brackets(text: str, start: int, end: int) -> Tuple[bool, bool]:
    # (inside, adjacent)
    s = start
    e = end
    inside = (s > 0 and text[s - 1] in "([") and (e < len(text) and text[e] in ")]")
    adjacent = (s > 0 and text[s - 1] in _LEADING_BRACK) or (e < len(text) and text[e] in _CLOSING_BRACK)
    return inside, adjacent

def _has_stands_for_near(text: str, start: int, end: int, radius: int) -> bool:
    lo = max(0, start - radius)
    hi = min(len(text), end + radius)
    return bool(_STANDS_FOR_RE.search(text[lo:hi]))

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
    while i < n and text[i] in "\"'“”‘’([{" :
        i += 1
    j = i
    while j < n and (text[j].isalpha() or text[j] == "'"):
        j += 1
    word = text[i:j]
    return bool(word) and word.islower()

def _normalize_key(surface: str) -> str:
    # 1) normalize apostrophes
    s = "".join(_APOSTROPHE_VARIANTS.get(ch, ch) for ch in surface)
    # 2) remove spaces *around* allowed separators only (R & D -> R&D), keep other spaces intact (rare).
    parts: List[str] = []
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

def _core_len_for_bounds(token: str) -> int:
    # count alnum only for min/max length checks
    return sum(1 for ch in token if ch.isalnum())

def _blacklist_context_drop(surface: str, text: str, start: int, end: int, cfg: DetectorConfig) -> bool:
    tok = surface
    if tok not in cfg.blacklist:
        return False
    # special: "IT" used as pronoun at sentence start: "IT was ..."
    if tok == "IT" and _is_sentence_start(text, start) and _next_word_lowercase(text, end):
        return True
    # "AM" after "I " (I AM ...) isn't an acronym
    if tok == "AM":
        # look behind for "I "
        i = start - 1
        while i >= 0 and text[i].isspace():
            i -= 1
        if i >= 0 and text[i] == 'I':
            # ensure before I is boundary-ish
            j = i - 1
            while j >= 0 and text[j].isspace():
                j -= 1
            if j < 0 or text[j] in ".!?\n\r\"'“”‘’([{":
                return True
    # Plain blacklist: if sentence-start and next word is lowercase, very likely not acronym (e.g., OK then)
    if _is_sentence_start(text, start) and _next_word_lowercase(text, end):
        return True
    return False

def _score(surface: str, text: str, start: int, end: int, cfg: DetectorConfig) -> float:
    # base
    score = 0.6

    inside, adjacent = _in_brackets(text, start, end)
    if inside:
        score += 0.25
    elif adjacent:
        score += 0.15

    if _has_stands_for_near(text, start, end, radius=40):
        score += 0.15

    if surface in cfg.blacklist:
        score -= 0.2

    # Clip
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score

def _context_window(text: str, start: int, end: int, window_chars: int) -> Tuple[int, int]:
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

    return (left, right)

def _iter_candidates(text: str, cfg: DetectorConfig) -> Iterator[Tuple[str, int, int]]:
    pat = _compile_pattern(cfg)
    for m in pat.finditer(text):
        s, e = m.span("tok")
        s, e = _strip_trailing_punct(text, s, e)
        if e - s < cfg.min_len:  # quick guard
            continue
        surface = text[s:e]
        # bounds by core alnum length
        clen = _core_len_for_bounds(surface)
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


# ---------------------------
# Public API
# ---------------------------

DEFAULT_CONFIG = DetectorConfig()


def detect_acronyms(text: str, config: DetectorConfig = DEFAULT_CONFIG) -> DetectorResult:
    """
    One-pass detector. Returns stable schema + first-occurrence map with normalized keys.
    """
    occurrences: List[Occurrence] = []
    firsts: Dict[str, FirstOccurrence] = {}

    for surface, s, e in _iter_candidates(text, config):
        if _blacklist_context_drop(surface, text, s, e, config):
            continue

        conf = _score(surface, text, s, e, config)
        ctx = _context_window(text, s, e, config.window_chars)

        occ = Occurrence(
            acronym=surface,
            start_offset=s,
            end_offset=e,
            confidence=conf,
            context_window=ctx,
        )
        occurrences.append(occ)

        key = _normalize_key(surface)
        if key not in firsts:
            firsts[key] = FirstOccurrence(
                acronym=surface,
                start_offset=s,
                end_offset=e,
                confidence=conf,
            )

    return DetectorResult(unique_acronyms=firsts, occurrences=occurrences)
