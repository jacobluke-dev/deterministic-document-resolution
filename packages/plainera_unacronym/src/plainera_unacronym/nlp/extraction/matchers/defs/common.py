import re
from dataclasses import dataclass
from typing import Optional, Literal, Callable

from plainera_unacronym.nlp.common.constants_regex import PUNCT_TRIM
from plainera_unacronym.nlp.common.shared import strip_trailing_punct_str, collapse_ws
from plainera_unacronym.nlp.extraction.matchers.common import is_mixed_case_acronym, split_compound


class LocalDefMatch:
    def __init__(self, def_start: int, def_end: int, definition: str, raw: str | None = None):
        self.def_start = def_start
        self.def_end = def_end
        self.definition = definition
        self.raw = raw


@dataclass(frozen=True, slots=True)
class InitialsStream:
    """
    `letters` is a list of scan-order initials (uppercased),
    `owners[i]` is the token index that produced `letters[i]`, and
    `is_stop[i]` is the stopword status of the owning token.
    """
    letters: list[str]
    owners: list[int]
    is_stop: list[bool]


@dataclass(frozen=True, slots=True)
class AlignmentHit:
    """
    `used_letter_pos`: indices into `stream.letters` used by the match,
    `hit_tokens`: set of token indices that contributed initials,
    `tok_left`/`tok_right`: inclusive token-span bounds covering `hit_tokens`.
    """
    used_letter_pos: list[int]
    hit_tokens: set[int]
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
    """Build a scan-order initials stream over `tokens`.

    The returned stream is the canonical representation used by acronym alignment:
    a flat list of initials (always uppercased) plus metadata mapping each initial
    back to its owning token and that token's stopword status.

    Scan direction affects both:
      * token traversal order, and
      * per-token part traversal order when `split_compounds=True`.

    Behaviour by option:
      * `treat_acronym_tokens_as_multi_letter`: acronym-like tokens (e.g. "U.S.A", "HTTP")
        contribute multiple letters rather than a single initial.
      * `expand_allcaps_tokens`: ALLCAPS alphabetic tokens contribute multiple letters.
      * Otherwise: each token contributes one initial per part, where parts are either:
          - `split_compound(tok)` if `split_compounds=True`, or
          - the token itself (as a single part).

    Args:
        tokens: Token strings (typically whitespace tokens from a snippet window).
        stopwords: Lowercased stopword set. from config
        scan: scanning direction either "ltr" or "rtl"
        expand_allcaps_tokens: If True, expand ALLCAPS alpha tokens (len > 1) into
            multiple letters (in scan order).
        split_compounds: If True, split compound tokens (hyphen/slash/dot/&/CamelCase,
            depending on `split_compound` implementation) and take initials per part.
        treat_acronym_tokens_as_multi_letter: If True, treat acronym-like tokens as
            multi-letter sources (e.g. "U.S.A" yields U,S,A).
    Returns:
        An `InitialsStream`
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
    lowercase_prefix_exception: bool,
) -> bool:
    """Return True if a leading lowercase acronym char may map to token[0].

    This is a narrow exception used for mixed-case acronyms such as "mRNA" or "iOS".
    It permits the *first* acronym character (acr_pos == 0) to align to the *first*
    token (token_idx == 0) even if that token is not a stopword, but only when the
    token genuinely appears to begin with that lowercase prefix.

    Args:
        acr: Acronym string being aligned (maybe mixed-case).
        tokens: Token list for the alignment window.
        token_idx: Candidate token index the acronym char would align to.
        acr_pos: Candidate acronym position being aligned.
        lowercase_prefix_exception: Feature flag enabling this exception.

    Returns:
        True if the exception applies for this `(acr_pos, token_idx)` pair, else False.

    Notes:
        The exception is disabled when:
          * `lowercase_prefix_exception` is False,
          * `acr_pos != 0` or `token_idx != 0`,
          * `acr` is not mixed-case (per `is_mixed_case_acronym`), or
          * token[0] is a pure ALLCAPS alphabetic token.

        Matching is performed by comparing the first character of token[0] and `acr`
        case-insensitively after trimming punctuation via `PUNCT_TRIM`.
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
    """Align an acronym string against an `InitialsStream`.

        This function aligns acronym characters (letters and/or numeric designators,
        depending on `acr_alignment_targets`) to the stream's scan-order initials, and
        returns token-span metadata describing which tokens contributed.

        Two alignment modes are supported:
          * `"rtl_scan"`: scan-based matching from RTL over `stream.letters`.
          * `"ltr_min_window"`: LTR matching that prefers a minimal token window
            (delegated to `_align_ltr_min_window`).

        Stopword constraints are enforced by the underlying aligner:
          * Uppercase acr letters typically must align to non-stopword tokens, unless
            `allow_upper_on_stop=True`.
          * Lowercase acronym letters may align to stopword tokens; if
            `allow_lower_on_non_stop=True` they may also align to non-stopword tokens.

        Args:
            acr: Acronym string to align. If empty, returns None.
            stream: Precomputed initials stream over the candidate token window.
                If `stream.letters` is empty, returns None.
            tokens: Original tokens used to build the stream; used for stopword status and
                any mode-specific heuristics.
            stopwords: Lowercased stopword set for determining token stopword status.
            mode: Alignment strategy selector: `"rtl_scan"` or `"ltr_min_window"`.
            allow_upper_on_stop: If True, permit uppercase acronym letters to land on
                stopword tokens.
            allow_lower_on_non_stop: If True, permit lowercase acronym letters to land on
                non-stopword tokens (useful for mixed-case acronyms).
            lowercase_prefix_exception: If True, enable a narrow exception allowing a
                leading lowercase acronym character (e.g. "mRNA") to map to token[0]
                when it looks like a true prefix (see `_lowercase_prefix_ok`).

        Returns:
            An `AlignmentHit` if alignment succeeds, else None.

        """
    if not acr or not stream.letters:
        return None

    # Preflight (shared)
    is_stop_token = [t.lower() in stopwords for t in tokens]
    has_num = has_numeric_evidence(tokens)
    Align_letters = acr_alignment_targets(acr, has_numeric_evidence=has_num)
    if not Align_letters:
        return None

    if mode == "rtl_scan":
        return _align_rtl_scan(
            Align_letters,
            stream=stream,
            allow_upper_on_stop=allow_upper_on_stop,
            allow_lower_on_non_stop=allow_lower_on_non_stop,
        )

    return _align_ltr_min_window(
        Align_letters,
        stream=stream,
        tokens=tokens,
        is_stop_token=is_stop_token,
        allow_upper_on_stop=allow_upper_on_stop,
        allow_lower_on_non_stop=allow_lower_on_non_stop,
        lowercase_prefix_exception=lowercase_prefix_exception,
    )


def _align_rtl_scan(
    acronym_alignment: list[str],
    *,
    stream: InitialsStream,
    allow_upper_on_stop: bool,
    allow_lower_on_non_stop: bool,
) -> Optional[AlignmentHit]:
    """Align acronym targets to the stream using right-to-left scanning.

        This is a thin wrapper around `align_rtl_scan(...)` that converts the matched
        stream-letter positions into token-span metadata (`AlignmentHit`).

        Args:
            acronym_alignment: Acronym alignment targets (typically from `acr_alignment_targets`), in the
                order expected by the RTL scan aligner.
            stream: Initials stream providing `letters`, `owners`, and per-letter stopword
                status `is_stop`.
            allow_upper_on_stop: If True, permit uppercase targets to match initials owned
                by stopword tokens.
            allow_lower_on_non_stop: If True, permit lowercase targets to match initials
                owned by non-stopword tokens.

        Returns:
            An `AlignmentHit` if a match is found, else None. The hit's token bounds are
            derived from the owning tokens of the matched stream positions.
        """
    used = align_rtl_scan(
        acronym_alignment, stream.letters, stream.is_stop,
        allow_upper_on_stop=allow_upper_on_stop,
        allow_lower_on_non_stop=allow_lower_on_non_stop,
    )
    if used is None:
        return None

    used = list(used)
    hit_tokens = {stream.owners[p] for p in used}
    return AlignmentHit(
        used_letter_pos=used,
        hit_tokens=hit_tokens,
        tok_left=min(hit_tokens),
        tok_right=max(hit_tokens),
    )


def _align_ltr_min_window(
    alignment_letters: list[str],
    *,
    stream: InitialsStream,
    tokens: list[str],
    is_stop_token: list[bool],
    allow_upper_on_stop: bool,
    allow_lower_on_non_stop: bool,
    lowercase_prefix_exception: bool,
) -> Optional[AlignmentHit]:
    L = [c.upper() for c in alignment_letters]  # stream letters are uppercase

    best_used: Optional[list[int]] = None
    best_span: Optional[tuple[int, int]] = None  # (tok_left, tok_right)

    for li in range(len(stream.letters)):
        if (len(stream.letters) - li) < len(L):
            break

        ai = 0
        used_letter_pos: list[int] = []

        for lj in range(li, len(stream.letters)):
            if stream.letters[lj] != L[ai]:
                continue

            tok_idx = stream.owners[lj]
            want_stop = alignment_letters[ai].islower()

            # --- stopword / case constraints ---
            if not want_stop:
                ok = (not is_stop_token[tok_idx]) or allow_upper_on_stop
            else:
                if is_stop_token[tok_idx]:
                    ok = True
                elif not allow_lower_on_non_stop:
                    ok = False
                else:
                    # narrow exception: mixed-case leading lowercase (mRNA / iOS)
                    if not lowercase_prefix_exception:
                        ok = False
                    elif ai != 0 or tok_idx != 0:
                        ok = False
                    else:
                        tok0 = tokens[0]
                        if tok0.isalpha() and tok0.isupper() and len(tok0) > 1:
                            ok = False
                        else:
                            ok = tok0[:1].lower() == alignment_letters[0].lower()

            if not ok:
                continue

            used_letter_pos.append(lj)
            ai += 1

            if ai == len(L):
                tok_left = min(stream.owners[p] for p in used_letter_pos)
                tok_right = max(stream.owners[p] for p in used_letter_pos)

                if best_span is None or (tok_right - tok_left) < (best_span[1] - best_span[0]):
                    best_span = (tok_left, tok_right)
                    best_used = list(used_letter_pos)
                break

    if not best_used:
        return None

    hit_tokens = {stream.owners[p] for p in best_used}
    tok_left, tok_right = best_span  # type: ignore[assignment]
    return AlignmentHit(
        used_letter_pos=best_used,
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
    False for: (Long Form), (ACR), (ACR, )
    """
    acr_esc = re.escape(acr)
    Q = r"""["'“”‘’]"""
    QUOTE = rf"(?:\s*{Q}\s*)?"

    return bool(re.match(
        # word boundary must apply to the acronym, not the closing quote
        rf"\(\s*{QUOTE}{acr_esc}\b{QUOTE}\s*"
        # delimiter introducing the tail
        rf"[,;:—–-]\s*"
        # tail must start with a real char, not whitespace / ')'
        rf"[^)\s]",
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


def align_rtl_scan(
    targets: list[str],
    initials: list[str],
    is_stop_letter: list[bool],
    *,
    allow_upper_on_stop: bool,
    allow_lower_on_non_stop: bool = False,
) -> list[int] | None:
    """
    Match `targets` RIGHT->LEFT against `initials` scanned in stream order.

    - `targets` are caseful characters from `acr_alignment_targets` (e.g. ['m','R','N','A'])
    - `initials` are uppercase initials in scan order
    - `is_stop_letter[i]` is stopword-status of the owner token for `initials[i]`

    Returns indices into `initials` used for the match, or None if not fully matched.
    """
    ti = len(targets) - 1  # target index (RTL)
    li = 0  # initials scan index
    used: list[int] = []

    while li < len(initials) and ti >= 0:
        need = targets[ti].upper()

        if initials[li] == need:
            want_stop = targets[ti].islower()

            if want_stop:
                ok = is_stop_letter[li] or allow_lower_on_non_stop
            else:
                ok = (not is_stop_letter[li]) or allow_upper_on_stop

            if ok:
                used.append(li)
                ti -= 1

        li += 1

    return None if ti >= 0 else used


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


def kept_token_indices(
    tokens: list[str],
    *,
    tok_left: int,
    tok_right: int,
    hit_tokens: set[int],
    bridges: set[str],
    include_numeric_leading: bool,
    first_alnum_char_upper: Callable[[str], str],
) -> list[int]:
    def _numeric_leading(idx: int) -> bool:
        if not include_numeric_leading:
            return False
        init = first_alnum_char_upper(tokens[idx])
        return (init is not None) and (not init.isalpha())

    kept = [
        idx
        for idx in range(tok_left, tok_right + 1)
        if idx in hit_tokens or tokens[idx].lower() in bridges or _numeric_leading(idx)
    ]
    return kept or list(range(tok_left, tok_right + 1))


def phrase_from_indices(tokens: list[str], idxs: list[int]) -> str:
    return strip_trailing_punct_str(collapse_ws(" ".join(tokens[i] for i in idxs)))


def build_kept_phrase(
    tokens: list[str],
    *,
    tok_left: int,
    tok_right: int,
    hit_tokens: set[int],
    bridges: set[str],
    include_numeric_leading: bool = True,
    first_alnum_char_upper: Callable[[str], str],
) -> str:
    def _numeric_leading(idx: int) -> bool:
        if not include_numeric_leading:
            return False
        init = first_alnum_char_upper(tokens[idx])
        return (init is not None) and (not init.isalpha())

    kept: list[str] = []
    for idx in range(tok_left, tok_right + 1):
        tok = tokens[idx]
        if idx in hit_tokens or tok.lower() in bridges or _numeric_leading(idx):
            kept.append(tok)

    if not kept:
        kept = tokens[tok_left: tok_right + 1]

    return strip_trailing_punct_str(collapse_ws(" ".join(kept)))
