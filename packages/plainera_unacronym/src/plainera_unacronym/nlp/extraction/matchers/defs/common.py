from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from plainera_unacronym.nlp.common.constants_regex import BRIDGES_DEFAULT, DEFAULT_STOPWORDS, PUNCT_TRIM
from plainera_unacronym.nlp.common.shared import collapse_ws, strip_trailing_punct_str
from plainera_unacronym.nlp.common.types import Span, as_str_set
from plainera_unacronym.nlp.extraction import ExtractionConfig
from plainera_unacronym.nlp.extraction.matchers.common import is_mixed_case_acronym, split_compound

_SEG_RE = re.compile(
    r"""
    [A-Z]+(?=[A-Z][a-z]) |   # "HTTP" in "HTTPServer" style
    [A-Z]?[a-z]+         |   # "La", "port", "electronic"
    [A-Z]+               |   # "OS"
    [0-9]+                   # digits
    """,
    re.VERBOSE,
)


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


def get_cfg_consts(cfg: ExtractionConfig, max_char_default: int = 80) -> tuple[set[str], set[str], int]:
    """
    Return `(bridges, stop, max_chars)` from `cfg` with sensible defaults.

    Coerces `cfg.bridges` (or `BRIDGES_DEFAULT`) and `cfg.stop`/`cfg.stopwords` (or `DEFAULT_STOPWORDS`)
    into concrete `set[str]` to avoid `set|frozenset|Any` leakage in callers and keep mypy happy.

    Args:
        cfg (ExtractionConfig): Extraction configuration object.
        max_char_default (int): Default maximum phrase length to use when
            `cfg.max_phrase_chars` is not present. Defaults to 80.

    Returns:
        tuple[set[str], set[str], int]:
            - `bridges`: Normalised set of bridge tokens.
            - `stop`: Normalised set of stopwords.
            - `max_chars`: Maximum phrase length.
    """
    bridges: set[str] = as_str_set(getattr(cfg, "bridges", None), default=BRIDGES_DEFAULT)

    _stop_raw = getattr(cfg, "stop", None)
    if _stop_raw is None:
        _stop_raw = getattr(cfg, "stopwords", None)

    stop: set[str] = as_str_set(_stop_raw, default=DEFAULT_STOPWORDS)

    max_chars = getattr(cfg, "max_phrase_chars", max_char_default)
    return bridges, stop, max_chars


def _acr_signature_for_initials(acr: str) -> str:
    """
    Build an initials-style acronym used ONLY for alignment against long-form initials.

    - Split into CamelCase / ALLCAPS segments
    - Emit:
        * all letters for ALLCAPS segments (OS -> O,S)
        * first letter for mixed/lower segments (TeX -> T, Bay -> B, f -> F)
    """
    segs = _SEG_RE.findall("".join(ch for ch in acr if ch.isalnum()))
    if not segs:
        return acr

    out: list[str] = []
    for seg in segs:
        if seg.isalpha() and seg.isupper() and len(seg) > 1:
            out.extend(list(seg))  # OS -> O,S
        else:
            out.append(seg[0].upper())  # TeX -> T, f -> F, Bay -> B, La -> L

    return "".join(out) or acr


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

        if treat_acronym_tokens_as_multi_letter and _try_emit_acronym_like_token(
            tok_clean,
            ti,
            scan=scan,
            is_stop=is_stop_tok[ti],
            letters=letters,
            owners=owners,
            is_stop_letter=is_stop_letter,
        ):
            continue

        if expand_allcaps_tokens and _try_emit_allcaps_token(
            tok_clean,
            ti,
            scan=scan,
            is_stop=is_stop_tok[ti],
            letters=letters,
            owners=owners,
            is_stop_letter=is_stop_letter,
        ):
            continue

        _emit_normal_initials(
            tok_clean,
            ti,
            scan=scan,
            split_compounds=split_compounds,
            is_stop=is_stop_tok[ti],
            letters=letters,
            owners=owners,
            is_stop_letter=is_stop_letter,
        )

    return InitialsStream(letters=letters, owners=owners, is_stop=is_stop_letter)


def _append_letters(
    chs: list[str],
    ti: int,
    *,
    is_stop: bool,
    letters: list[str],
    owners: list[int],
    is_stop_letter: list[bool],
) -> None:
    """Append letters to the initials stream buffers with ownership metadata.

    Args:
        chs: Letters to append (already in scan order).
        ti: Owning token index.
        is_stop: Whether the owning token is a stopword.
        letters: Output letters buffer.
        owners: Output token-owner index buffer (parallel to `letters`).
        is_stop_letter: Output stopword flag buffer (parallel to `letters`).

    Returns:
        None
    """
    for ch in chs:
        letters.append(ch.upper())
        owners.append(ti)
        is_stop_letter.append(is_stop)


def _try_emit_acronym_like_token(
    tok_clean: str,
    ti: int,
    *,
    scan: Literal["ltr", "rtl"],
    is_stop: bool,
    letters: list[str],
    owners: list[int],
    is_stop_letter: list[bool],
) -> bool:
    """Emit multi-letter initials for acronym-like tokens.

    Uses `_acronym_letters_rtl` to derive letters and then converts into scan order.

    Args:
        tok_clean: Token text with outer punctuation trimmed.
        ti: Token index in the original token list.
        scan: Scanning direction ("ltr" or "rtl").
        is_stop: Whether the token is a stopword.
        letters: Output letters buffer.
        owners: Output token-owner index buffer.
        is_stop_letter: Output stopword flag buffer.

    Returns:
        bool: True if token was handled (letters emitted), else False.
    """
    if not is_acronym_like_token(tok_clean):
        return False

    # _acronym_letters_rtl returns RTL order; convert to scan order.
    chs = list(_acronym_letters_rtl(tok_clean))
    if scan == "ltr":
        chs = list(reversed(chs))

    if not chs:
        return False

    _append_letters(chs, ti, is_stop=is_stop, letters=letters, owners=owners, is_stop_letter=is_stop_letter)
    return True


def _try_emit_allcaps_token(
    tok_clean: str,
    ti: int,
    *,
    scan: Literal["ltr", "rtl"],
    is_stop: bool,
    letters: list[str],
    owners: list[int],
    is_stop_letter: list[bool],
) -> bool:
    """Emit multi-letter initials for ALLCAPS alphabetic tokens when enabled.

    Args:
        tok_clean: Token text with outer punctuation trimmed.
        ti: Token index in the original token list.
        scan: Scanning direction ("ltr" or "rtl").
        is_stop: Whether the token is a stopword.
        letters: Output letters buffer.
        owners: Output token-owner index buffer.
        is_stop_letter: Output stopword flag buffer.

    Returns:
        bool: True if token was handled (letters emitted), else False.
    """
    if not (tok_clean.isalpha() and tok_clean.isupper() and len(tok_clean) > 1):
        return False

    chs = list(tok_clean)
    if scan == "rtl":
        chs = list(reversed(chs))

    _append_letters(chs, ti, is_stop=is_stop, letters=letters, owners=owners, is_stop_letter=is_stop_letter)
    return True


def _emit_normal_initials(
    tok_clean: str,
    ti: int,
    *,
    scan: Literal["ltr", "rtl"],
    split_compounds: bool,
    is_stop: bool,
    letters: list[str],
    owners: list[int],
    is_stop_letter: list[bool],
) -> None:
    """Emit initials for the standard (non-multi-letter) path.

    Splits the token into parts if `split_compounds=True`, then emits the first alnum
    character per part (in scan order).

    Args:
        tok_clean: Token text with outer punctuation trimmed.
        ti: Token index in the original token list.
        scan: Scanning direction ("ltr" or "rtl").
        split_compounds: Whether to split compound tokens into parts.
        is_stop: Whether the token is a stopword.
        letters: Output letters buffer.
        owners: Output token-owner index buffer.
        is_stop_letter: Output stopword flag buffer.

    Returns:
        None
    """
    parts = split_compound(tok_clean) if split_compounds else [tok_clean]
    if not parts:
        return

    part_iter = parts if scan == "ltr" else reversed(parts)
    for part in part_iter:
        ch = first_alnum_char_upper(part)
        if ch is None:
            continue
        letters.append(ch.upper())
        owners.append(ti)
        is_stop_letter.append(is_stop)


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
) -> AlignmentHit | None:
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
    mixed = bool(acr) and is_mixed_case_acronym(acr)
    if mixed:
        # If the acronym has more letters than the candidate token window can possibly explain,
        # fall back to the "caps skeleton" (e.g. LaTeX->LTX, eBay->EB).
        alpha_len = sum(c.isalpha() for c in acr)
        if tokens and alpha_len > len(tokens) or lowercase_prefix_exception and acr[0].islower():
            acr = _acr_signature_for_initials(acr)

    if acr and lowercase_prefix_exception and is_mixed_case_acronym(acr):
        acr = _acr_signature_for_initials(acr)

    if not acr or not stream.letters:
        return None

    # Preflight (shared)
    is_stop_token = [t.lower() in stopwords for t in tokens]
    has_num = has_numeric_evidence(tokens)
    Align_letters = acr_alignment_targets(acr, has_numeric_evidence=has_num)
    if not Align_letters:
        return None

    if mode == "rtl_scan":
        return _align_rtl_scan_wrapper(
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


def _align_rtl_scan_wrapper(
    acronym_alignment: list[str],
    *,
    stream: InitialsStream,
    allow_upper_on_stop: bool,
    allow_lower_on_non_stop: bool,
) -> AlignmentHit | None:
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
        acronym_alignment,
        stream.letters,
        stream.is_stop,
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
) -> AlignmentHit | None:
    """Align acronym letters to a initials stream using a minimal token-span strategy.

    Scans the stream left-to-right and tries to match `alignment_letters` in order.
    Among all valid matches, selects the one that minimises `(tok_right - tok_left)`
    where token bounds are derived from `stream.owners`.

    Case/stopword constraints:
      - Uppercase target letters prefer non-stopword tokens unless `allow_upper_on_stop`.
      - Lowercase target letters prefer stopword tokens; mapping to non-stopwords is
        only allowed when `allow_lower_on_non_stop`, and can be further restricted
        by `lowercase_prefix_exception` for mixed-case acronyms (e.g. iOS/mRNA).

    Args:
        alignment_letters: Acronym letters to align (may include lowercase to signal
            stopword preference). Must be alphanumeric-only upstream.
        stream: Prebuilt initials stream (letters are expected uppercase).
        tokens: Original token list the stream was derived from.
        is_stop_token: Parallel list to `tokens` indicating stopword status per token.
        allow_upper_on_stop: If True, allow uppercase target letters to align onto
            stopword tokens.
        allow_lower_on_non_stop: If True, allow lowercase target letters to align onto
            non-stopword tokens (subject to `lowercase_prefix_exception`).
        lowercase_prefix_exception: If True, allow a narrow exception for a leading
            lowercase letter in a mixed-case acronym to map to token0.

    Returns:
        An `AlignmentHit` describing the chosen alignment, or `None` if no valid
        alignment exists.
    """
    L = [c.upper() for c in alignment_letters]  # stream letters are uppercase

    best_used: list[int] | None = None
    best_span: Span | None = None  # (tok_left, tok_right)

    for li in range(len(stream.letters)):
        if (len(stream.letters) - li) < len(L):
            break

        used = _try_align_from_ltr_start(
            alignment_letters,
            L,
            li,
            stream=stream,
            tokens=tokens,
            is_stop_token=is_stop_token,
            allow_upper_on_stop=allow_upper_on_stop,
            allow_lower_on_non_stop=allow_lower_on_non_stop,
            lowercase_prefix_exception=lowercase_prefix_exception,
        )
        if not used:
            continue

        tok_left, tok_right = _token_span_for_used(used, stream.owners)

        if best_span is None or (tok_right - tok_left) < (best_span[1] - best_span[0]):
            best_span = (tok_left, tok_right)
            best_used = used

    if not best_used or best_span is None:
        return None

    hit_tokens = {stream.owners[p] for p in best_used}
    tok_left, tok_right = best_span
    return AlignmentHit(
        used_letter_pos=best_used,
        hit_tokens=hit_tokens,
        tok_left=tok_left,
        tok_right=tok_right,
    )


def _try_align_from_ltr_start(
    alignment_letters: list[str],
    L: list[str],
    li: int,
    *,
    stream: InitialsStream,
    tokens: list[str],
    is_stop_token: list[bool],
    allow_upper_on_stop: bool,
    allow_lower_on_non_stop: bool,
    lowercase_prefix_exception: bool,
) -> list[int] | None:
    """Attempt to align from a specific LTR starting stream index.

    Scans `stream.letters[li:]` and tries to match the acronym letters `L` in order.
    Applies stopword/case constraints per letter. Returns the used stream positions
    for the first valid completion (minimal end position for this `li`).

    Args:
        alignment_letters: Original acronym letters (may include lowercase).
        L: Uppercased version of `alignment_letters`.
        li: Starting index into `stream.letters`.
        stream: Prebuilt initials stream.
        tokens: Original token list.
        is_stop_token: Stopword flags per token.
        allow_upper_on_stop: Allow uppercase letters to hit stop tokens.
        allow_lower_on_non_stop: Allow lowercase letters to hit non-stop tokens.
        lowercase_prefix_exception: Allow narrow exception for mixed-case leading lowercase.

    Returns:
        list[int] | None: Used stream letter positions if a full match is found, else None.
    """
    ai = 0
    used_letter_pos: list[int] = []

    for lj in range(li, len(stream.letters)):
        if stream.letters[lj] != L[ai]:
            continue

        tok_idx = stream.owners[lj]
        if not _alignment_letter_ok(
            alignment_letters,
            ai,
            tok_idx,
            tokens=tokens,
            is_stop_token=is_stop_token,
            allow_upper_on_stop=allow_upper_on_stop,
            allow_lower_on_non_stop=allow_lower_on_non_stop,
            lowercase_prefix_exception=lowercase_prefix_exception,
        ):
            continue

        used_letter_pos.append(lj)
        ai += 1

        if ai == len(L):
            return used_letter_pos

    return None


def _alignment_letter_ok(
    alignment_letters: list[str],
    ai: int,
    tok_idx: int,
    *,
    tokens: list[str],
    is_stop_token: list[bool],
    allow_upper_on_stop: bool,
    allow_lower_on_non_stop: bool,
    lowercase_prefix_exception: bool,
) -> bool:
    """Check whether a single aligned letter may map to a given token.

    Implements the stopword/case constraints described in `_align_ltr_min_window`.

    Args:
        alignment_letters: Original acronym letters (may include lowercase).
        ai: Index into `alignment_letters` for the letter being matched.
        tok_idx: Candidate token index being hit.
        tokens: Original token list.
        is_stop_token: Stopword flags per token.
        allow_upper_on_stop: Allow uppercase letters to hit stop tokens.
        allow_lower_on_non_stop: Allow lowercase letters to hit non-stop tokens.
        lowercase_prefix_exception: Allow narrow exception for mixed-case leading lowercase.

    Returns:
        bool: True if mapping is allowed, False otherwise.
    """
    want_stop = alignment_letters[ai].islower()

    if not want_stop:
        return (not is_stop_token[tok_idx]) or allow_upper_on_stop

    # want stopword
    if is_stop_token[tok_idx]:
        return True
    if not allow_lower_on_non_stop:
        return False

    # narrow exception: mixed-case leading lowercase (mRNA / iOS)
    if not lowercase_prefix_exception or ai != 0 or tok_idx != 0:
        return False

    tok0 = tokens[0]
    if tok0.isalpha() and tok0.isupper() and len(tok0) > 1:
        return False

    return tok0[:1].lower() == alignment_letters[0].lower()


def _token_span_for_used(used_letter_pos: list[int], owners: list[int]) -> Span:
    """Compute (tok_left, tok_right) from used stream letter positions.

    Args:
        used_letter_pos: Stream indices used in an alignment.
        owners: Parallel list mapping stream indices to owning token index.

    Returns:
        Span: (tok_left, tok_right) inclusive token bounds.
    """
    tok_left = min(owners[p] for p in used_letter_pos)
    tok_right = max(owners[p] for p in used_letter_pos)
    return tok_left, tok_right


def expand_numeric_leading_window(
    tokens: list[str],
    tok_left: int,
    tok_right: int,
) -> Span:
    """Expand a token window to include adjacent numeric-leading tokens.

    A token is treated as numeric-leading if its first alphanumeric character
    exists and is not a letter (e.g., "3M", "2", "10GbE").

    Args:
        tokens: Token sequence that the window refers to.
        tok_left: Inclusive left token index of the current window.
        tok_right: Inclusive right token index of the current window.

    Returns:
        A tuple ``(new_left, new_right)`` representing the expanded inclusive window.
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
    Check whether `snippet` is a parenthetical containing `acr` plus a trailing “tail”.
    True for: (ACR, ...), ("ACR": ...), ('ACR' - ...)
    False for: (Long Form), (ACR), (ACR, )
    Args:
        snippet: The parenthetical text to classify (typically including the surrounding parentheses).
        acr: The acronym expected to appear at the start of the parenthetical content.

    Returns:
        True if `snippet` matches the acronym-parenthetical-with-tail pattern; otherwise False.

    """
    acr_esc = re.escape(acr)
    Q = r"""["'“”‘’]"""
    QUOTE = rf"(?:\s*{Q}\s*)?"

    return bool(
        re.match(
            # word boundary must apply to the acronym, not the closing quote
            rf"\(\s*{QUOTE}{acr_esc}\b{QUOTE}\s*"
            # delimiter introducing the tail
            rf"[,;:—–-]\s*"
            # tail must start with a real char, not whitespace / ')'
            rf"[^)\s]",
            snippet,
            flags=re.UNICODE,
        )
    )


# Hard clause boundary for inline defs (stop scanning / gating at these)
_INLINE_BOUNDARY_RE = re.compile(r"[.;:](?=\s|$)|[\r\n]")


def inline_clause_tail(s: str) -> tuple[str, int]:
    """Return the leading clause fragment before any hard boundary.

    A “hard boundary” is matched by `_INLINE_BOUNDARY_RE` (e.g. `. ; :` at
    token/line ends, or a newline). If no boundary is found, the full string is returned.

    Returns:
        (tail_text, tail_end_index) where `tail_text == s[:tail_end_index]`.
    """
    m = _INLINE_BOUNDARY_RE.search(s)
    end = m.start() if m else len(s)
    return s[:end], end


def first_alnum_char_upper(s: str) -> str | None:
    """Return the first alphanumeric character in `s`, uppercased.

    Returns None if `s` contains no alphanumeric characters.
    """
    for ch in s:
        if ch.isalnum():
            return ch.upper()
    return None


def acr_alignment_targets(acr: str, *, has_numeric_evidence: bool) -> list[str]:
    """Return acronym chars to align (alnum only), optionally dropping digits.

    If `has_numeric_evidence` is True, keep digits (e.g., "10GbE").
    Otherwise drop digits so acronyms like "E2E" can match "end-to-end".

    Args:
        acr: Acronym text (may include punctuation).
        has_numeric_evidence: Whether the candidate phrase indicates digits matter.

    Returns:
        Alignment target characters, or [] if `acr` has no alnum chars.
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
    """True if any token starts (by first alnum char) with a non-letter.

    Uses `first_alnum_char_upper(token)` and checks `not init.isalpha()`.

    Args:
        tokens: Candidate phrase tokens.

    Returns:
        Whether the phrase contains numeric-leading evidence.
    """
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
    """Align acronym targets right-to-left against an initials stream scanned in order.

    This performs a greedy scan over `initials` from left to right, while consuming
    `targets` from right to left. A target character matches when the uppercased
    target equals the current initials letter and stopword constraints are satisfied.

    Stopword constraints:
      - Lowercase target (e.g. 'o' in "MofM") prefers a stopword owner letter. It may
        align to non-stop letters only if `allow_lower_on_non_stop=True`.
      - Uppercase target prefers a non-stop owner letter. It may align to stop letters
        only if `allow_upper_on_stop=True`.

    Args:
        targets: Acronym alignment targets (caseful). Lowercase indicates a stopword
            preference; uppercase indicates a non-stopword preference.
        initials: Initials letters in scan order (expected uppercase).
        is_stop_letter: Parallel to `initials`; True if the owning token is a stopword.
        allow_upper_on_stop: If True, allow uppercase targets to align onto stop letters.
        allow_lower_on_non_stop: If True, allow lowercase targets to align onto non-stop letters.

    Returns:
        A list of indices into `initials` representing the matched letters (in scan order),
        or None if the full target sequence cannot be matched.
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
    """Return True if `tok` looks like an acronym/initialism.

    Treats tokens as acronym-like if, after trimming common trailing punctuation,
    they match dotted/compact patterns such as "U.S.A" or contain uppercase letters
    (and optional digits) with no lowercase letters (e.g. "HTTP2", "RNA").

    Args:
        tok: Raw token text, possibly with trailing punctuation.

    Returns:
        True if the token resembles an acronym/initialism; otherwise False.
    """
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
    """Extract acronym characters in right-to-left order.

    Strips light trailing/leading punctuation, keeps only alphanumeric characters,
    uppercases letters, and returns them in RTL order.

    Args:
        tok: Token potentially containing an acronym/initialism (e.g. "U.S.A", "HTTP2").

    Returns:
        List of uppercased alphanumeric characters in RTL order.
        Examples:
          - "RNA" -> ["A", "N", "R"]
          - "U.S.A" -> ["A", "S", "U"]
          - "HTTP2" -> ["2", "P", "T", "T", "H"]
    """
    t = tok.strip(".,;:)]}»”'\"")
    chars = [c.upper() for c in t if c.isalnum()]
    chars.reverse()
    return chars


def strip_inline_cue_prefix(snippet: str, cfg) -> tuple[str, int] | None:
    """Strips a leading inline cue phrase from text.

    This detects whether `snippet` starts with one of `cfg.inline_cues` (case-insensitive),
    allowing optional leading whitespace and an optional leading comma, then
    requires at least one whitespace character after the cue.

    Args:
        snippet: Input text to test/strip.
        cfg: Config object providing `inline_cues` (iterable of regex fragments).

    Returns:
        A tuple of (`remaining_text`, `offset`) where `remaining_text` is `t` with the
        matched prefix removed, and `offset` is the character index into `t` where the
        remaining text begins. Returns None if no cue matches at the start.
    """
    cues = getattr(cfg, "inline_cues", ())
    for cue in cues:
        m = re.match(rf"^\s*,?\s*(?:{cue})\s+", snippet, flags=re.IGNORECASE)
        if m:
            return snippet[m.end() :], m.end()
    return None


def _numeric_leading(token, include_numeric_leading: bool) -> bool:
    """
    Returns True if `token` should be treated as numeric-leading.

    A token is numeric-leading when `include_numeric_leading` is True and the
    first alphanumeric character in `token` exists and is NOT a letter
    (e.g. starts with a digit like "3M", "10GbE", "(2FA)").

    Args:
        token: Raw token text (may include surrounding punctuation/whitespace).
        include_numeric_leading: Feature-flag; when False this always returns False.

    Returns:
        True if numeric-leading tokens should be included and `token` begins
        (after skipping non-alnum) with a non-alpha alnum character; otherwise False.
    """
    if not include_numeric_leading:
        return False
    init = first_alnum_char_upper(token)
    return (init is not None) and (not init.isalpha())


def kept_token_indices(
    tokens: list[str],
    *,
    tok_left: int,
    tok_right: int,
    hit_tokens: set[int],
    bridges: set[str],
    include_numeric_leading: bool,
) -> list[int]:
    """
    Select token indices to retain for a rendered definition phrase.

    Keeps:
      - tokens that directly contributed to the acronym alignment (`hit_tokens`)
      - "bridge" tokens that improve readability (e.g. {"of", "and"})
      - numeric-leading tokens when enabled (e.g. "3M", "2", "10GbE")

    If nothing qualifies, falls back to the full contiguous window
    `[tok_left .. tok_right]` to avoid returning an empty selection.

    Args:
        tokens: Token list for the candidate definition span.
        tok_left: Inclusive left bound of the matched token window.
        tok_right: Inclusive right bound of the matched token window.
        hit_tokens: Token indices that contributed initials used in the match.
        bridges: Lowercased connector tokens that should be preserved for readability.
        include_numeric_leading: If True, include numeric-leading tokens inside the window.

    Returns:
        A list of indices (ascending) within `[tok_left..tok_right]` to keep.

    Raises:
        None.
    """
    kept = [
        idx
        for idx in range(tok_left, tok_right + 1)
        if idx in hit_tokens or tokens[idx].lower() in bridges or _numeric_leading(tokens[idx], include_numeric_leading)
    ]
    return kept or list(range(tok_left, tok_right + 1))


def phrase_from_indices(tokens: list[str], idxs: list[int]) -> str:
    """
    Build a display phrase from selected token indices.

    Joins `tokens[i]` for `i` in `idxs`, collapses internal whitespace, then strips
    any trailing punctuation/whitespace (per `strip_trailing_punct_str`).

    Args:
        tokens: Source token list.
        idxs: Indices into `tokens` to include, in the desired output order.

    Returns:
        The rendered phrase string (may be empty if `idxs` is empty or selected tokens collapse away).

    Raises:
        IndexError: If any index in `idxs` is out of range for `tokens`.
    """
    return strip_trailing_punct_str(collapse_ws(" ".join(tokens[i] for i in idxs)))


def build_kept_phrase(
    tokens: list[str],
    *,
    tok_left: int,
    tok_right: int,
    hit_tokens: set[int],
    bridges: set[str],
    include_numeric_leading: bool = True,
) -> str:
    """
    Build a display phrase for a matched token window.

    Keeps tokens in the inclusive window `[tok_left..tok_right]` if they are:
    - contributing hit tokens (`idx in hit_tokens`), or
    - bridge tokens (`tokens[idx].lower() in bridges`), or
    - numeric-leading tokens (when `include_numeric_leading=True`).

    If nothing qualifies, falls back to keeping the full window.
    Output is whitespace-collapsed and has trailing punctuation stripped.

    Args:
        tokens: Full token list for the candidate phrase.
        tok_left: Leftmost token index (inclusive) of the candidate window.
        tok_right: Rightmost token index (inclusive) of the candidate window.
        hit_tokens: Token indices that directly contributed to the acronym match.
        bridges: Lowercased bridge words to keep for readability (e.g. {"of", "and"}).
        include_numeric_leading: Whether numeric-leading tokens (e.g. "3M", "2") are kept.

    Returns:
        A rendered phrase string (may be empty if the selected window contains only whitespace).

    Raises:
        IndexError: If `tok_left`/`tok_right` are out of range for `tokens`.
        ValueError: If `tok_left > tok_right`.
    """
    if tok_left > tok_right:
        raise ValueError("tok_left must be <= tok_right")

    # 1) Identify "core" tokens we *must* keep (hits + numeric-leading)
    core_idxs: list[int] = []
    for idx in range(tok_left, tok_right + 1):
        if idx in hit_tokens or _numeric_leading(tokens[idx], include_numeric_leading):
            core_idxs.append(idx)

    core_min = min(core_idxs) if core_idxs else None
    core_max = max(core_idxs) if core_idxs else None

    kept: list[str] = []
    for idx in range(tok_left, tok_right + 1):
        tok = tokens[idx]
        low = tok.lower()

        if idx in hit_tokens or _numeric_leading(tok, include_numeric_leading):
            kept.append(tok)
            continue

        # 2) Bridges: keep only if they sit strictly between two core tokens
        if core_min is not None and core_max is not None and core_min < idx < core_max and low in bridges:
            kept.append(tok)

    if not kept:
        kept = tokens[tok_left : tok_right + 1]

    return strip_trailing_punct_str(collapse_ws(" ".join(kept)))
