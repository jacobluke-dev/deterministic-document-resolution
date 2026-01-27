import re
from dataclasses import dataclass
from typing import Optional, Literal

from plainera_unacronym.nlp.common.constants_regex import PUNCT_TRIM
from plainera_unacronym.nlp.extraction.matchers.common import is_mixed_case_acronym, split_compound


class LocalDefMatch:
    def __init__(self, def_start: int, def_end: int, definition: str, raw: str | None = None):
        self.def_start = def_start
        self.def_end = def_end
        self.definition = definition
        self.raw = raw


@dataclass(frozen=True, slots=True)
class InitialsStream:
    letters: list[str]      # scan-order initials (already UPPER)
    owners: list[int]       # token index per letter
    is_stop: list[bool]     # per-letter stopword status (same len as letters)


@dataclass(frozen=True, slots=True)
class AlignmentHit:
    used_letter_pos: list[int]  # indices into stream.letters
    hit_tokens: set[int]        # token indices that contributed
    tok_left: int
    tok_right: int


def build_initials_stream(
    tokens: list[str],
    *,
    stopwords: set[str],
    scan: Literal["ltr", "rtl"],
    expand_allcaps_tokens: bool,
    split_compounds: bool,
    treat_acronym_tokens_as_multi_letter: bool,
) -> InitialsStream:
    """
    Build a canonical initials stream over tokens.

    - letters are always uppercased
    - owners maps each letter back to a token index
    - is_stop is per-letter stopword status derived from owning token

    Flags are explicit; no hidden policy.
    """
    letters: list[str] = []
    owners: list[int] = []
    is_stop_letter: list[bool] = []

    is_stop_tok = [t.lower() in stopwords for t in tokens]

    tok_indices = range(len(tokens)) if scan == "ltr" else range(len(tokens) - 1, -1, -1)

    for ti in tok_indices:
        tok = tokens[ti]
        tok_clean = tok.strip(PUNCT_TRIM)

        # Option: treat acronym-like tokens as multi-letter (e.g., "U.S.A" or "HTTP")
        if treat_acronym_tokens_as_multi_letter and is_acronym_like_token(tok_clean):
            # _acronym_letters_rtl returns RTL order; convert to scan order
            chs = list(_acronym_letters_rtl(tok_clean))
            if scan == "ltr":
                chs = list(reversed(chs))
            for ch in chs:
                letters.append(ch.upper())
                owners.append(ti)
                is_stop_letter.append(is_stop_tok[ti])
            continue

        # Option: expand ALLCAPS word tokens into multiple letters (only when you want it)
        if expand_allcaps_tokens and tok_clean.isalpha() and tok_clean.isupper() and len(tok_clean) > 1:
            chs = list(tok_clean)
            if scan == "rtl":
                chs = list(reversed(chs))
            for ch in chs:
                letters.append(ch.upper())
                owners.append(ti)
                is_stop_letter.append(is_stop_tok[ti])
            continue

        # Normal path: split compounds or take token as-is, then grab first alnum char per part
        parts = split_compound(tok_clean) if split_compounds else [tok_clean]
        if not parts:
            continue

        part_iter = parts if scan == "ltr" else reversed(parts)
        for part in part_iter:
            ch = first_alnum_char_upper(part)
            if ch is None:
                continue
            letters.append(ch.upper())
            owners.append(ti)
            is_stop_letter.append(is_stop_tok[ti])

    return InitialsStream(letters=letters, owners=owners, is_stop=is_stop_letter)


def _lowercase_prefix_ok(
    *,
    acr: str,
    tokens: list[str],
    token_idx: int,
    acr_pos: int,
    is_stop_token: list[bool],
    lowercase_prefix_exception: bool,
) -> bool:
    """
    Narrow exception: allow mixed-case leading lowercase (mRNA / iOS) to map to non-stopword token[0]
    only when it genuinely looks like a prefix char on token0.
    """
    if not lowercase_prefix_exception:
        return False
    if acr_pos != 0 or token_idx != 0:
        return False
    if not is_mixed_case_acronym(acr):
        return False

    tok0 = tokens[0].strip(PUNCT_TRIM)
    if tok0.isalpha() and tok0.isupper():
        return False
    return tok0[:1].lower() == acr[:1].lower()


def align_acronym_to_initials(
    acr: str,
    stream: InitialsStream,
    *,
    tokens: list[str],
    stopwords: set[str],
    mode: Literal["ltr_min_window", "rtl_scan"],
    allow_upper_on_stop: bool,
    allow_lower_on_non_stop: bool,
    lowercase_prefix_exception: bool,
) -> Optional[AlignmentHit]:
    """
    Canonical alignment:
      - rtl_scan: use your existing align(...) against stream in its scan order
      - ltr_min_window: choose smallest token window satisfying acronym letters in order

    Note: `stream.letters` are uppercase; `acr_alignment_targets` returns caseful chars.
    """
    if not acr or not stream.letters:
        return None

    is_stop_token = [t.lower() in stopwords for t in tokens]
    has_num = has_numeric_evidence(tokens)
    A = acr_alignment_targets(acr, has_numeric_evidence=has_num)
    if not A:
        return None

    if mode == "rtl_scan":
        # Use existing primitive (expects targets + letters + per-letter stop flags)
        used = align(
            A, stream.letters, stream.is_stop,
            allow_upper_on_stop=allow_upper_on_stop,
            allow_lower_on_non_stop=allow_lower_on_non_stop,
        )
        if used is None:
            return None

        hit_tokens = {stream.owners[p] for p in used}
        tok_left = min(hit_tokens)
        tok_right = max(hit_tokens)
        return AlignmentHit(
            used_letter_pos=list(used),
            hit_tokens=hit_tokens,
            tok_left=tok_left,
            tok_right=tok_right,
        )

    # ---- ltr_min_window ----
    # We want the smallest token window. We scan over stream letters (which are already in scan order).
    # This assumes the stream was built with scan="ltr". If you pass scan="rtl", you’ll get nonsense.
    L = [c.upper() for c in A]

    best: Optional[tuple[int, int, list[int]]] = None  # (tok_left, tok_right, used_letter_pos)

    for li in range(len(stream.letters)):
        ai = 0
        used_letter_pos: list[int] = []

        for lj in range(li, len(stream.letters)):
            if stream.letters[lj] != L[ai]:
                continue

            tok_idx = stream.owners[lj]
            want_stop = A[ai].islower()

            if not want_stop:
                ok = (not is_stop_token[tok_idx]) or allow_upper_on_stop
            else:
                ok = is_stop_token[tok_idx] or (
                    allow_lower_on_non_stop and _lowercase_prefix_ok(
                        acr=acr,
                        tokens=tokens,
                        token_idx=tok_idx,
                        acr_pos=ai,
                        is_stop_token=is_stop_token,
                        lowercase_prefix_exception=lowercase_prefix_exception,
                    )
                )

            if not ok:
                continue

            used_letter_pos.append(lj)
            ai += 1
            if ai == len(L):
                tok_left = min(stream.owners[p] for p in used_letter_pos)
                tok_right = max(stream.owners[p] for p in used_letter_pos)

                if best is None or (tok_right - tok_left) < (best[1] - best[0]):
                    best = (tok_left, tok_right, used_letter_pos)
                break

        # early exit: not enough remaining letters to finish the acronym
        if (len(stream.letters) - li) < len(L):
            break

    if not best:
        return None

    tok_left, tok_right, used = best
    hit_tokens = {stream.owners[p] for p in used}
    return AlignmentHit(
        used_letter_pos=list(used),
        hit_tokens=hit_tokens,
        tok_left=tok_left,
        tok_right=tok_right,
    )


def expand_numeric_leading_window(
    tokens: list[str],
    tok_left: int,
    tok_right: int,
) -> tuple[int, int]:
    """
    Expand [tok_left..tok_right] to include adjacent numeric-leading tokens (e.g. "3M", "2").
    """
    def _numeric_leading(idx: int) -> bool:
        init = first_alnum_char_upper(tokens[idx])
        return (init is not None) and (not init.isalpha())

    while tok_left > 0 and _numeric_leading(tok_left - 1):
        tok_left -= 1
    while tok_right + 1 < len(tokens) and _numeric_leading(tok_right + 1):
        tok_right += 1

    return tok_left, tok_right

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
