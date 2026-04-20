from __future__ import annotations

import re
from typing import Literal

from document_resolution.nlp.common.constants_regex import BRIDGES_DEFAULT, DEFAULT_STOPWORDS, PUNCT_TRIM
from document_resolution.nlp.common.shared import collapse_ws, strip_trailing_punct_str
from document_resolution.nlp.common.types import Span, as_str_set
from document_resolution.nlp.extraction.acronyms.config import ExtractionConfig
from document_resolution.nlp.extraction.acronyms.matchers.common import is_mixed_case_acronym, split_compound
from document_resolution.nlp.extraction.acronyms.matchers.defs.constants import InitialsStream, AlignmentHit
from document_resolution.nlp.extraction.acronyms.matchers.defs.ltr import _align_ltr_min_window
from document_resolution.nlp.extraction.acronyms.matchers.defs.rtl import _acronym_letters_rtl, _align_rtl_scan_wrapper

_SEG_RE = re.compile(
    r"""
    [A-Z]+(?=[A-Z][a-z]) |   # "HTTP" in "HTTPServer" style
    [A-Z]?[a-z]+         |   # "La", "port", "electronic"
    [A-Z]+               |   # "OS"
    [0-9]+                   # digits
    """,
    re.VERBOSE,
)

type scanner = Literal["ltr", "rtl"]


def get_cfg_consts(cfg: ExtractionConfig, max_char_default: int = 80) -> tuple[set[str], set[str], int]:
    """
    Return `(bridges, stop, max_chars)` from `cfg` with sensible defaults.

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
    scan: scanner,
    expand_allcaps_tokens: bool,
    split_compounds: bool,
    treat_acronym_tokens_as_multi_letter: bool,
) -> InitialsStream:
    """Build a scan-order initials stream over `tokens`.

    The returned stream is the canonical representation used by acronym alignment:
    a flat list of initials (always uppercased) plus metadata mapping each initial
    back to its owning token and that token's stopword status.

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
    """
    for ch in chs:
        letters.append(ch.upper())
        owners.append(ti)
        is_stop_letter.append(is_stop)


def _try_emit_acronym_like_token(
    tok_clean: str,
    ti: int,
    *,
    scan: scanner,
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
    scan: scanner,
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
    scan: scanner,
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


def expand_numeric_leading_window(
    tokens: list[str],
    tok_left: int,
    tok_right: int,
) -> Span:
    """Expand a token window to include adjacent numeric-leading tokens.
    """

    def _is_numeric_leading_token(idx: int) -> bool:
        init = first_alnum_char_upper(tokens[idx])
        return (init is not None) and (not init.isalpha())

    while tok_left > 0 and _is_numeric_leading_token(tok_left - 1):
        tok_left -= 1
    while tok_right + 1 < len(tokens) and _is_numeric_leading_token(tok_right + 1):
        tok_right += 1

    return tok_left, tok_right


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
