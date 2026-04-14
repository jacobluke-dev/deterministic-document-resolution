from __future__ import annotations

import re
from collections.abc import Iterator

from document_resolution.nlp.common.constants_regex import (
    CLOSING_BRACK,
    DEFAULT_TWO_LETTER_BOOST,
    LEADING_BRACK,
    STANDS_FOR_RE,
    TIME_RE,
    TRAILING_PUNCT_CHARS,
)
from document_resolution.nlp.common.shared import has_letter, has_paren_definition, normalize_acronym_key
from document_resolution.nlp.common.types import AcronymDetectorConfig, Span, TextSpanTuple
from document_resolution.nlp.plugins.registry import DOMAIN_PLUGINS

_DOTTED_INITIALISM_RE = re.compile(r"^(?:[A-Z]\.)+[A-Z]$")


def letters(token: str) -> str:
    """
    Extract alphabetic characters from a token surface.

    This is used for casing calculations where punctuation and digits should not
    affect ratios. The relative order of letters is preserved.

    Args:
        token (str): Token surface to filter.

    Returns:
        str: Letters-only version of the token.
    """
    return "".join(ch for ch in token if ch.isalpha())


def caps_ratio(token: str) -> float:
    ls = letters(token)
    if not ls:
        return 0
    upp = sum(1 for ch in ls if ch.isupper())
    return upp / len(ls)


def strip_trailing_punct_span(text: str, start: int, end: int) -> Span:
    """
    Trim common trailing punctuation from a candidate span.

    This adjusts offsets without allocating substrings and helps avoid matching
    tokens like 'NHS,' or 'U.S.A.)' with punctuation included.

    Args:
        text (str): Source text.
        start (int): Start offset (inclusive).
        end (int): End offset (exclusive).

    Returns:
        Span: Adjusted (start, end) with trailing punctuation removed.
    """
    while end > start and text[end - 1] in TRAILING_PUNCT_CHARS:
        end -= 1
    return start, end


def in_brackets(text: str, start: int, end: int) -> tuple[bool, bool]:
    """
    Determine whether a span is bracketed or bracket-adjacent.

    The 'inside' flag requires both an opening bracket immediately before and a
    closing bracket immediately after. The 'adjacent' flag triggers if either side
    touches any configured bracket character.

    Args:
        text (str): Source text.
        start (int): Start offset (inclusive).
        end (int): End offset (exclusive).

    Returns:
        tuple[bool, bool]: (inside, adjacent) flags.
    """
    s = start
    e = end
    inside = (s > 0 and text[s - 1] in "([") and (e < len(text) and text[e] in ")]")
    adjacent = (s > 0 and text[s - 1] in LEADING_BRACK) or (e < len(text) and text[e] in CLOSING_BRACK)
    return inside, adjacent


def has_stands_for_follow(text: str, end: int, max_chars: int = 24) -> bool:
    """
    Detect a right-hand "stands for" cue near a candidate.

    Scans forward from `end`, skipping whitespace and stopping at sentence-ending
    punctuation or `max_chars`, then applies STANDS_FOR_RE to that slice.

    Args:
        text (str): Source text.
        end (int): End offset (exclusive) of the candidate span.
        max_chars (int): Maximum lookahead length. Defaults to 24.

    Returns:
        bool: True if a "stands for" cue is detected to the right; else False.
    """
    i, n = end, len(text)
    while i < n and text[i].isspace():
        i += 1
    j = i
    limit = min(n, i + max_chars)
    while j < limit and text[j] not in ".!?":
        j += 1
    return bool(STANDS_FOR_RE.search(text[i:j]))


def next_word_lowercase(text: str, end: int) -> bool:
    """
    Check whether the next lexical word after a span is all lowercase.

    This is a cheap heuristic to detect patterns like "NHS is ..." where the token
    behaves like a normal noun phrase boundary. Quotes/brackets are skipped.

    Args:
        text (str): Source text.
        end (int): End offset (exclusive) of the candidate span.

    Returns:
        bool: True if the next word exists and is lowercase; else False.
    """
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
    """
    Extract the immediately preceding token to the left of a span.

    Scans backwards over whitespace then collects a simple alnum/":." token.
    Used for lightweight context checks (e.g., time markers).

    Args:
        text (str): Source text.
        start (int): Start offset (inclusive) of the candidate span.

    Returns:
        str: Previous token surface (may be empty).
    """
    i = start - 1
    while i >= 0 and text[i].isspace():
        i -= 1
    j = i
    while j >= 0 and (text[j].isalnum() or text[j] in ":."):
        j -= 1
    return text[j + 1 : i + 1]


def core_len_for_bounds(token: str) -> int:
    """
    Compute alphanumeric length for min/max gating.

    This ignores punctuation/separators so tokens like "R&D" are evaluated on their
    alnum content, consistent with other acceptance heuristics.

    Args:
        token (str): Token surface.

    Returns:
        int: Count of .isalnum() characters in the token.
    """
    return sum(1 for ch in token if ch.isalnum())


def threshold_len(surface: str, allow_chars: str) -> int:
    """
    Compute effective length used for confidence/threshold heuristics.

    Base length is the alnum count. If any allowed separator (or a dot) exists, the
    effective length is boosted to at least 3 so forms like "R&D" are not treated as
    low-signal two-character tokens.

    Args:
        surface (str): Matched surface text.
        allow_chars (str): Allowed internal separators (e.g., "&/-").

    Returns:
        int: Effective length used in threshold-based logic.
    """
    clen = core_len_for_bounds(surface)
    if any(ch in allow_chars for ch in surface) or "." in surface:
        return max(3, clen)
    return clen


def boost_confidence_if_whitelisted(surface: str, confidence_score: float, cfg: AcronymDetectorConfig) -> float:
    """
    Boost confidence for whitelisted two-letter keys.

    Normalizes the surface to a key and, if the key is exactly two letters and
    present in cfg.whitelist_two_letter, applies a bounded additive boost.

    Args:
        surface (str): Candidate surface text.
        confidence_score (float): Current confidence score.
        cfg (AcronymDetectorConfig): Detector configuration (whitelist and boost settings).

    Returns:
        float: Updated confidence score (capped at 0.99).
    """
    key = normalize_acronym_key(
        surface,
        allow_chars=cfg.allow_chars,
        dotted_mode=cfg.dotted_display,
    )

    if len(key) == 2 and key in cfg.whitelist_two_letter:
        boost = getattr(cfg, "two_letter_boost", DEFAULT_TWO_LETTER_BOOST)
        return min(confidence_score + boost, 0.99)
    return confidence_score


def calc_score(surface: str, text: str, start: int, end: int, cfg: AcronymDetectorConfig) -> float:
    """
    Compute a confidence score for an accepted candidate using local cues.

    The score starts from a baseline and is adjusted by bracket placement, nearby
    definition cues, "stands for" cues, and blacklist membership.

    Args:
        surface (str): Candidate surface text.
        text (str): Source text.
        start (int): Start offset (inclusive).
        end (int): End offset (exclusive).
        cfg (AcronymDetectorConfig): Detector configuration (blacklists, etc.).

    Returns:
        float: Confidence score clamped to [0.0, 1.0].
    """
    score = 0.6
    inside, adjacent = in_brackets(text, start, end)
    if inside:
        score += 0.25
    elif adjacent:
        score += 0.15
    if has_paren_definition(text, end):
        score += 0.25
    if has_stands_for_follow(text, end):
        score += 0.15
    if surface in cfg.blacklist:
        score -= 0.2
    return max(0.0, min(1.0, score))


def context_window(text: str, start: int, end: int, window_chars: int) -> Span:
    """
    Build a bounded sentence-like context window around a span.

    Expands left/right until a terminator (., !, ?, newline) or `window_chars` is reached,
    then trims leading whitespace on the left and includes the terminator on the right.

    Args:
        text (str): Source text.
        start (int): Span start offset (inclusive).
        end (int): Span end offset (exclusive).
        window_chars (int): Maximum expansion distance on each side.

    Returns:
        Span: (left, right) offsets delimiting the context window.
    """
    # Left: back to previous terminator (or start), then skip spaces
    left = start
    while left > 0 and text[left - 1] not in ".!?\n\r":
        if start - left >= window_chars:
            break
        left -= 1
    while left < start and text[left].isspace():
        left += 1

    # Right: forward to next terminator (or end), include the terminator
    n = len(text)
    right = end
    while right < n and text[right] not in ".!?\n\r":
        if right - end >= window_chars:
            break
        right += 1
    if right < n and text[right] in ".!?\n\r":
        right += 1  # include the terminator

    return left, right


def _has_lower_and_upper(tok: str) -> bool:
    """
    Check whether a token contains both lowercase and uppercase letters.

    Used to relax caps-ratio requirements for mixed-case forms (e.g., mRNA, eBPF)
    when cfg.enable_mixed_case is enabled.

    Args:
        tok (str): Token surface.

    Returns:
        bool: True if both cases occur among alphabetic characters; else False.
    """
    return any(c.islower() for c in tok if c.isalpha()) and any(c.isupper() for c in tok if c.isalpha())


def _is_lower_prefix_brand(surface: str) -> bool:
    # eBay, iOS, xAPI, mDNS, etc.
    # 1–2 lowercase prefix, then exactly 1 uppercase, then at least 1 more alnum
    # (keeps it narrow; avoids normal words)
    return bool(re.match(r"^[a-z]{1,2}[A-Z][A-Za-z0-9]+$", surface))


def _accept_candidate(text: str, cfg: AcronymDetectorConfig, s: int, e: int) -> TextSpanTuple | None:
    """
    Apply standard gating to a raw (s, e) match and return an accepted span.

    Performs trailing punctuation stripping, letter presence checks, length bounds,
    dotted-initialism validation, and caps-ratio gating (with mixed-case relaxation).

    Args:
        text (str): Source text.
        cfg (AcronymDetectorConfig): Detector configuration (bounds, allowlists, ratios).
        s (int): Start offset (inclusive).
        e (int): End offset (exclusive).

    Returns:
        TextSpanTuple | None: (surface, s, e) if accepted; otherwise None.
    """
    s, e = strip_trailing_punct_span(text, s, e)
    if e - s < cfg.min_len:
        return None

    surface = text[s:e]
    if not has_letter(surface):
        return None

    if not _passes_dotted_gates(text, cfg, surface, s, e):
        return None

    if not _passes_generic_gates(cfg, surface):
        return None

    return surface, s, e


def _passes_dotted_gates(text: str, cfg: AcronymDetectorConfig, surface: str, s: int, e: int) -> bool:
    """
    Validate dotted-initialism constraints when '.' appears in the surface.

    This performs validation only (does not mutate surface). It ensures dotted tokens
    are clean initialisms (e.g., "U.S", "U.S.A") and applies additional context guards
    to avoid common false positives (section numbers, dotted chains, etc.).

    Args:
        text (str): Source text for boundary/context checks.
        cfg (AcronymDetectorConfig): Detector configuration (length bounds, allowlists).
        surface (str): Candidate surface text (already punctuation-stripped).
        s (int): Start offset (inclusive) of the candidate in `text`.
        e (int): End offset (exclusive) of the candidate in `text`.

    Returns:
        bool: True if dotted gates pass (or not applicable); False otherwise.
    """
    if "." not in surface:
        return True

    # Must be a clean dotted initialism like U.S or U.S.A (trailing '.' already stripped).
    if not _DOTTED_INITIALISM_RE.fullmatch(surface):
        return False

    letters_only = surface.replace(".", "")

    # Length checks should use letters_only (same as core_len_for_bounds, but explicit here).
    if len(letters_only) < cfg.min_len or len(letters_only) > cfg.max_len:
        return False

    # 2-letter dotted is too noisy unless whitelisted (US/UK/EU/UN etc.).
    if len(letters_only) == 2 and letters_only not in cfg.whitelist_two_letter:
        return False

    # Context guards: avoid picking up section numbers / weird dotted chains.
    if s > 0 and text[s - 1].isdigit():
        return False
    if e < len(text) and text[e].isdigit():
        return False
    if s > 0 and text[s - 1] == ".":
        return False

    # If immediately followed by '.' that's OK only for common "U.S.A.)" / "U.S.A.," patterns.
    if e < len(text) and text[e] == ".":
        nxt = text[e + 1] if e + 1 < len(text) else ""
        if nxt and nxt not in ")]}»”'\" \n\r\t,;:!?…":
            # e.g. "U.S.A.X" should be rejected
            return False

    return True


def _passes_generic_gates(cfg: AcronymDetectorConfig, surface: str) -> bool:
    """
    Apply generic (non-dotted-specific) gating to a candidate surface.

    This includes core-length bounds, hard caps (e.g. >=15, ==1), and caps-ratio gating.
    For mixed-case surfaces, caps-ratio requirement may be relaxed when there are at
    least two uppercase letters.

    Args:
        cfg (AcronymDetectorConfig): Detector configuration (length bounds and ratio thresholds).
        surface (str): Candidate surface text (already punctuation-stripped).

    Returns:
        bool: True if the generic gates pass; False otherwise.
    """
    _LOWER_PREFIX_BRAND_RE = re.compile(r"^[a-z]{1,2}[A-Z][A-Za-z0-9]*$")

    clen = core_len_for_bounds(surface)

    # hard guards
    if clen < cfg.min_len or clen >= 15 or clen == 1:
        return False

    enable_mixed = getattr(cfg, "enable_mixed_case", False)

    # max_len is enforced for everyone *except* a small mixed-case allowance
    if clen > cfg.max_len and not (
        enable_mixed
        and _has_lower_and_upper(surface)  # true mixed-case only
        and clen <= max(cfg.max_len, 6)  # tiny spillover only
    ):
        return False

    req = cfg.require_caps_ratio
    if enable_mixed and _has_lower_and_upper(surface):
        upp = sum(1 for ch in surface if ch.isalpha() and ch.isupper())

        if upp >= 2:
            req = min(req, cfg.require_caps_ratio_mixed)

        # NEW: allow lower-prefix brand tokens like eBay/iOS-style when only 1 upper
        elif upp == 1 and _LOWER_PREFIX_BRAND_RE.match(surface):
            # Keep this conservative: brand tokens should not need 0.7 caps ratio.
            # 0.25 is enough for eBay (1/4). If you want stricter, use 0.3.
            req = min(req, 0.25)

    return caps_ratio(surface) >= req


def _collect_core_hits(text: str, cfg: AcronymDetectorConfig, pat: re.Pattern[str]) -> list[TextSpanTuple]:
    """
    Collect accepted core-regex hits in text order.

    Iterates over `pat.finditer` using the named group "tok" and applies the same
    acceptance gating as the legacy candidate path.

    Args:
        text (str): Source text.
        cfg (AcronymDetectorConfig): Detector configuration.
        pat (re.Pattern[str]): Compiled pattern containing group "tok".

    Returns:
        list[TextSpanTuple]: Accepted hits as (surface, start, end) tuples.
    """
    out: list[TextSpanTuple] = []
    for m in pat.finditer(text):
        s, e = m.span("tok")  # our pattern's named group
        hit = _accept_candidate(text, cfg, s, e)
        if hit:
            out.append(hit)
    return out


def _collect_domain_hits(text: str, cfg: AcronymDetectorConfig) -> list[TextSpanTuple]:
    """Collect accepted domain-plugin hits and sort for containment checks.

    Sorted by (start asc, length desc) so longer domain spans come first.
    """
    hits: list[TextSpanTuple] = []
    for name in cfg.enabled_domains or ():
        plug = DOMAIN_PLUGINS.get(name)
        if not plug:
            continue
        for _, s, e in plug.extra_candidates(text, cfg) or ():
            hit = _accept_candidate(text, cfg, s, e)
            if hit:
                hits.append(hit)

    hits.sort(key=lambda x: (x[1], -(x[2] - x[1])))
    return hits


def _contained_in_any(s: int, e: int, containers: list[TextSpanTuple]) -> bool:
    """
    Check whether a span is fully contained within any container span.

    Containers are expected to be sorted by start offset; the function uses an
    early-exit when container starts surpass the target end.

    Args:
        s (int): Start offset (inclusive) of the target span.
        e (int): End offset (exclusive) of the target span.
        containers (list[TextSpanTuple]): Candidate container spans.

    Returns:
        bool: True if (s, e) is contained in any container; else False.
    """
    for _, ds, de in containers:
        if ds <= s and e <= de:
            return True
        if ds > e:
            break  # early exit: later spans all start after (s,e)
    return False


def iter_acronym_candidates(text: str, cfg: AcronymDetectorConfig, pat: re.Pattern[str]) -> Iterator[TextSpanTuple]:
    """
    Yield accepted candidates while suppressing obvious fragments.

    Core regex hits are emitted first unless fully contained by any domain span.
    Domain plugin hits are then emitted, with duplicates removed by (start, end).

    Args:
        text (str): Source text.
        cfg (AcronymDetectorConfig): Detector configuration.
        pat (re.Pattern[str]): Compiled core token pattern.

    Yields:
        TextSpanTuple: Accepted candidates in output order (core-first, then domain).
    """
    core_hits: list[TextSpanTuple] = _collect_core_hits(text, cfg, pat)
    dom_hits: list[TextSpanTuple] = _collect_domain_hits(text, cfg)

    seen: set[Span] = set()

    # 1) Core first (unless contained by any domain span)
    for c in core_hits:
        _, s, e = c
        core_key: Span = (s, e)
        if core_key in seen:
            continue
        if _contained_in_any(s, e, dom_hits):
            continue
        seen.add(core_key)
        yield c

    # 2) Then domain hits (skip duplicates by offsets)
    for h in dom_hits:
        _, s, e = h
        dom_key: Span = (s, e)
        if dom_key in seen:
            continue
        seen.add(dom_key)
        yield h


def reason_tags(surface: str, text: str, start: int, end: int, cfg: AcronymDetectorConfig) -> list[str]:  # noqa: C901
    """
    Derive lightweight “reason” tags for a matched acronym span, based on local
    context and config. Tags highlight cues that may boost or penalize confidence.

    Tag rules (added in this order when true):
      - "inside_parens"         → span is fully inside (...) or [...]
      - "adjacent_parens"       → span touches a bracket/paren boundary
      - "paren_definition_right"→ a parenthetical definition immediately follows
      - "stands_for_right"      → “stands for …” pattern appears to the right
      - "soft_blacklist_penalty"→ surface is in cfg.soft_blacklist
      - "non_acronym_upper"     → surface is in cfg.non_acronym_upper
      - "next_word_lowercase"   → next lexical word after `end` starts lowercase
      - "prev_time_token"       → previous token matches TIME_RE (e.g. “8PM”)
      - "has_separator"         → surface contains any character from cfg.allow_chars
      - "dotted_initialism"     → surface contains '.' and cfg.enable_dotted is True

    Args:
        surface: The matched surface text (typically text[start:end]).
        text: Full source text.
        start: Start offset (inclusive) of the match in `text`.
        end: End offset (exclusive) of the match in `text`.
        cfg: Detector configuration (uses `soft_blacklist`, `non_acronym_upper`,
             `allow_chars`, and `enable_dotted`).

    Returns:
        list[str]: Ordered list of tags describing contextual/lexical cues.

    Notes:
        - Bracket/parenthesis checks come from `in_brackets(text, start, end)`.
        - Right-hand cues use `has_paren_definition(text, end)` and
          `has_stands_for_follow(text, end)`.
        - Lowercase-next-word uses `next_word_lowercase(text, end)`.
        - The previous token is obtained via `prev_token(text, start)` and tested
          against `TIME_RE`.
        - Separator and dotted checks are purely character-based on `surface`.
    """
    inside, adjacent = in_brackets(text, start, end)
    prev = prev_token(text, start)

    is_inside_parens = inside
    is_adjacent_parens = (not inside) and adjacent
    has_paren_def_right = has_paren_definition(text, end)
    has_stands_for = has_stands_for_follow(text, end)
    is_blacklisted = surface in cfg.blacklist
    is_non_acronym_upper = surface in cfg.non_acronym_upper
    next_word_is_lower = next_word_lowercase(text, end)
    prev_is_time = bool(TIME_RE.match(prev))
    has_sep = any(ch in cfg.allow_chars for ch in surface)
    is_dotted_init = ("." in surface) and cfg.enable_dotted

    candidates = [
        ("inside_parens", is_inside_parens),
        ("adjacent_parens", is_adjacent_parens),
        ("paren_definition_right", has_paren_def_right),
        ("stands_for_right", has_stands_for),
        ("blacklist_penalty", is_blacklisted),
        ("non_acronym_upper", is_non_acronym_upper),
        ("next_word_lowercase", next_word_is_lower),
        ("prev_time_token", prev_is_time),
        ("has_separator", has_sep),
        ("dotted_initialism", is_dotted_init),
    ]
    return [tag for tag, ok in candidates if ok]
