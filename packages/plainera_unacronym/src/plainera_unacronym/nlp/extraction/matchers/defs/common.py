import re


class LocalDefMatch:
    def __init__(self, def_start: int, def_end: int, definition: str, raw: str | None = None):
        self.def_start = def_start
        self.def_end = def_end
        self.definition = definition
        self.raw = raw


def is_acronym_parenthetical_with_tail(snippet: str, acr: str) -> bool:
    """
    True for: (ACR, ...), ("ACR": ...), ('ACR' - ...)
    False for: (Long Form), (ACR)
    """
    acr_esc = re.escape(acr)
    Q = r"""["'“”‘’]"""
    QUOTE = rf"(?:\s*{Q}\s*)?"

    return bool(re.match(
        rf"\(\s*{QUOTE}{acr_esc}{QUOTE}\b\s*[,;:—–-]\s*\S",
        snippet,
        flags=re.UNICODE,
    ))


# Hard clause boundary for inline defs (stop scanning / gating at these)
_INLINE_BOUNDARY_RE = re.compile(r"[.;:](?=\s|$)|[\r\n]")


def inline_clause_tail(s: str) -> tuple[str, int]:
    """
    Return (tail_text, tail_end_index) where tail is from start of `s` up to
    the first hard boundary, or full `s` if none.
    """
    m = _INLINE_BOUNDARY_RE.search(s)
    end = m.start() if m else len(s)
    return s[:end], end


def first_alnum_char_upper(s: str) -> str | None:
    for ch in s:
        if ch.isalnum():
            return ch.upper()
    return None


def acr_alignment_targets(acr: str, *, has_numeric_evidence: bool) -> list[str]:
    """
    Alignment targets for acronym matching.
    If the candidate definition has no numeric-leading token evidence, digits are treated as optional
    (i.e., dropped) so E2E can match "end-to-end".
    """
    chars = [c for c in acr if c.isalnum()]
    if not chars:
        return []

    # If the phrase provides numeric evidence (e.g., "3M", "10GbE"), keep digits.
    if has_numeric_evidence:
        return chars

    # Otherwise, match letters only (digits become optional)
    letters = [c for c in chars if c.isalpha()]
    return letters


def has_numeric_evidence(tokens: list[str]) -> bool:
    for tok in tokens:
        init = first_alnum_char_upper(tok)
        if init is not None and not init.isalpha():
            return True
    return False


def align(
    A,
    letters,
    part_is_stop,
    *,
    allow_upper_on_stop: bool,
    allow_lower_on_non_stop: bool = False,
) -> list[int] | None:
    j = len(A) - 1
    k = 0
    used: list[int] = []

    while k < len(letters) and j >= 0:
        need = A[j].upper()

        if letters[k] == need:
            want_stop = A[j].islower()

            if want_stop:
                # strict: lowercase must land on stopword
                # relaxed: allow it to land on non-stopword too (for mixed-case acronyms like mRNA)
                ok = part_is_stop[k] or allow_lower_on_non_stop
            else:
                ok = (not part_is_stop[k]) or allow_upper_on_stop

            if ok:
                used.append(k)
                j -= 1
        k += 1

    return None if j >= 0 else used


_ACR_TOKEN_RE = re.compile(r"^[A-Z](?:[A-Z0-9]|[A-Z]\.){1,}$")  # RNA, HTTP2, U.S.A


def is_acronym_like_token(tok: str) -> bool:
    # Trim light punctuation that commonly sticks to tokens
    t = tok.strip(".,;:)]}»”'\"")
    if len(t) < 2:
        return False
    # Common dotted initialisms: U.S.A
    if _ACR_TOKEN_RE.fullmatch(t):
        return True
    # Pure uppercase letters/digits (no lowercase) is also acronym-like
    return any(c.isupper() for c in t) and not any(c.islower() for c in t) and any(c.isalpha() for c in t)


def _acronym_letters_rtl(tok: str) -> list[str]:
    """
    Return alnum chars from token as UPPER, in RTL order.
    Example: 'RNA' -> ['A','N','R'], 'U.S.A' -> ['A','S','U'], 'HTTP2' -> ['2','P','T','T','H'] (digits kept).
    """
    t = tok.strip(".,;:)]}»”'\"")
    chars = [c.upper() for c in t if c.isalnum()]
    chars.reverse()
    return chars


def strip_inline_cue_prefix(t: str, cfg) -> tuple[str, int] | None:
    """
    If `t` begins with an inline cue ("stands for", "means", ...), return:
      - remaining text after the cue
      - number of characters stripped (offset into original `t`)
    """
    cues = getattr(cfg, "inline_cues", ())
    for cue in cues:
        m = re.match(rf"^\s*,?\s*(?:{cue})\s+", t, flags=re.IGNORECASE)
        if m:
            return t[m.end():], m.end()
    return None
