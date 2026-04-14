import re

from document_resolution.nlp.common.constants_regex import QUOTE
from document_resolution.nlp.common.shared import collapse_ws, has_letter
from document_resolution.nlp.extraction.acronyms.core.normalise import normalize_definition
from document_resolution.nlp.extraction.acronyms.matchers.common import is_mixed_case_acronym
from document_resolution.nlp.extraction.acronyms.matchers.defs.common import (
    LocalDefMatch,
    align_acronym_to_initials,
    build_initials_stream,
    build_kept_phrase,
    expand_numeric_leading_window,
    first_alnum_char_upper,
    get_cfg_consts,
)
from document_resolution.nlp.extraction.acronyms.matchers.defs.inline_after import scan_tokens
from document_resolution.nlp.extraction.acronyms.matchers.numeric_matcher import consume_left_numeric_designator

_CAMEL_RE = re.compile(r"[a-z][A-Z]")


def _has_camelcase_token(tokens: list[str]) -> bool:
    return any(_CAMEL_RE.search(t.strip()) for t in tokens)


def requires_compound_split_for_alignment(
    acr: str,
    tokens: list[str],
    stopwords: set[str],
) -> bool:
    """
    Return True if intra-token compound splitting is required to make acronym
    alignment *possible* given the available token window.

    The check is purely arithmetic and deterministic:

      - Count the number of alphabetic characters in `acr`.
      - Count the number of tokens in `tokens` that can contribute a letter
        initial (i.e. not a stopword and whose first alphanumeric character
        is alphabetic).
      - If the acronym requires more letter hits than the window can supply
        via whole-token initials, compound splitting must be enabled or
        alignment cannot succeed.

    Numeric-leading tokens (e.g. "2") are excluded from the initial-bearing
    count because they cannot satisfy alphabetic acronym characters.

    This function does *not* inspect punctuation, CamelCase, or lexical
    overrides — it answers only the question:
        "Is splitting mathematically required for alignment to be possible?"

    Args:
        acr: Acronym surface form (may contain digits).
        tokens: Candidate token window immediately preceding the acronym.
        stopwords: Lowercased stopword set.

    Returns:
        bool: True if compound splitting is required for alignment to be
        feasible; otherwise False.

    Examples:
        >>> requires_compound_split_for_alignment(
        ...     "HTTP2",
        ...     ["Hypertext", "Transfer", "Protocol", "2"],
        ...     stopwords=set(),
        ... )
        True   # 4 letters required, only 3 initial-bearing tokens available

        >>> requires_compound_split_for_alignment(
        ...     "PDF",
        ...     ["Portable", "Document", "Format"],
        ...     stopwords=set(),
        ... )
        False  # 3 letters, 3 initial-bearing tokens
    """
    if not acr or not tokens:
        return False

    # Count alphabetic characters required by the acronym.
    alpha_len = sum(c.isalpha() for c in acr)
    if alpha_len == 0:
        return False

    # Count tokens that can contribute a letter initial.
    initial_bearing_tokens = sum(
        1 for t in tokens if t.lower() not in stopwords and (first_alnum_char_upper(t) or "").isalpha()
    )

    return alpha_len > initial_bearing_tokens


def needs_compound_split_for_parenthetical_before(
    acr: str,
    tokens: list[str],
    is_mixed: bool,
    *,
    stopwords: set[str],
) -> bool:
    """
    Decide whether compound splitting should be enabled for the
    `LongForm ... (ACR)` anchored matcher.

    Enables splitting when:
      - the acronym is mixed-case (more permissive initials behaviour),
      - tokens include common compound separators (- / & .),
      - tokens include CamelCase segments, or
      - splitting is mathematically required for alignment feasibility
        (see `requires_compound_split_for_alignment`).

    Args:
        acr: Acronym surface form.
        is_mixed: if the Acronym is mixed case.
        tokens: Candidate token window preceding the acronym.
        stopwords: Lowercased stopword set.

    Returns:
        bool: Whether to enable compound splitting for initials stream building.
    """
    mixed = bool(acr) and is_mixed

    has_separators = any(("-" in t) or ("/" in t) or ("&" in t) or ("." in t) for t in tokens)

    must_split_to_fit = requires_compound_split_for_alignment(acr, tokens, stopwords)

    return mixed or has_separators or _has_camelcase_token(tokens) or must_split_to_fit


def find_parenthetical_longform_before_acr(snippet: str, acr: str, cfg) -> list[LocalDefMatch]:  # noqa: C901
    """Find a long-form definition immediately before a parenthesised acronym.

    Matches the anchored pattern:

        Long Form ... (ACR)

    where the closing wrapper `(...ACR...)` must occur at the end of `snippet`.
    The long-form window is selected by scanning tokens right-to-left and aligning
    the acronym letters to an initials stream built from the candidate phrase.

    High-level behaviour:
      - Uses a regex look-ahead to capture the text immediately before `(ACR)` while
        allowing a small “tail” inside the wrapper such as `(ACR, ...)`.
      - Tokenises the candidate prefix left-to-right to preserve stable character spans,
        but performs matching right-to-left to prefer the nearest plausible definition.
      - Builds an initials stream that can split compounds (hyphens/slashes/dots/& and
        CamelCase) and optionally treat acronym-like tokens as multi-letter parts.
      - Attempts acronym-to-initials alignment with strict rules first, then retries
        with relaxed rules if needed.
      - Expands the chosen token window to include adjacent numeric-leading tokens and
        keeps matched tokens plus configured bridge words for readability.
      - Returns tight `(def_start, def_end)` spans into the original `snippet` along with
        a normalised display definition.

    Args:
        snippet: Text ending with a parenthesised acronym occurrence, e.g.
            `"Portable Document Format (PDF)."` The matcher is anchored to the end.
        acr: Acronym surface form to align (letters/digits supported).
        cfg: Extraction config-like object. Reads:
            - `max_phrase_chars` (default 80)
            - `stopwords` (default `DEFAULT_STOPWORDS`)
            - `bridges` (default `BRIDGES_DEFAULT`)

    Returns:
        A list of `LocalDefMatch`. Empty if no match is found. When present, the
        list contains a single best match with:
          - `def_start` / `def_end`: character offsets into `snippet`
          - `definition`: normalised definition string for display
          - `raw`: whitespace-collapsed raw window from `snippet[def_start:def_end]`

    Notes:
        - This function enforces `max_phrase_chars` on the raw pre-wrapper prefix
          before any tightening/normalisation.
        - Offsets are computed against the original `snippet`; normalisation does
          not alter indices.
    """
    bridges, stop, max_chars = get_cfg_consts(cfg)

    acr_esc = re.escape(acr)

    tail = r"(?:\s*[,;:]\s*[^)]{0,120})?"  # cap the tail to stay sane

    m = re.search(
        rf"(?P<pre>[^\(\)]{{1,{max_chars}}})\s*" rf"(?=\(\s*{QUOTE}{acr_esc}{QUOTE}{tail}\s*\)\s*$)",
        snippet,
    )

    if not m:
        return []

    pre = m.group("pre").rstrip()
    if not pre or not has_letter(pre):
        return []

    # Raw length guard (before any tightening): enforce the configured limit strictly.
    if len(pre) > max_chars:
        return []

    # 1) Tokenize LTR by whitespace to get stable spans
    tokens, starts, ends = scan_tokens(pre, offset=0)

    if not tokens:
        return []

    acr_starts_with_digit = acr and acr[0].isdigit()
    is_mixed = is_mixed_case_acronym(acr)
    mixed = bool(acr) and is_mixed

    needs_compound_split = needs_compound_split_for_parenthetical_before(
        acr,
        tokens,
        is_mixed,
        stopwords=stop,
    )

    stream = build_initials_stream(
        tokens,
        stopwords=stop,
        scan="rtl",
        expand_allcaps_tokens=mixed,
        split_compounds=needs_compound_split,
        treat_acronym_tokens_as_multi_letter=needs_compound_split,
    )

    hit = align_acronym_to_initials(
        acr,
        stream,
        tokens=tokens,
        stopwords=stop,
        mode="rtl_scan",
        allow_upper_on_stop=False,
        allow_lower_on_non_stop=is_mixed,
        lowercase_prefix_exception=False,  # don’t need it here usually
    )

    if hit is None:
        hit = align_acronym_to_initials(
            acr,
            stream,
            tokens=tokens,
            stopwords=stop,
            mode="rtl_scan",
            allow_upper_on_stop=True,
            allow_lower_on_non_stop=is_mixed,
            lowercase_prefix_exception=False,
        )
    if hit is None:
        return []

    tok_left = hit.tok_left
    tok_right = len(tokens) - 1  # anchored to end in this matcher

    if acr and acr[0].isdigit():
        tok_left = consume_left_numeric_designator(acr=acr, tokens=tokens, tok_left=tok_left)

    tok_left, tok_right = expand_numeric_leading_window(tokens, tok_left, tok_right)

    hit_tokens = hit.hit_tokens

    if acr_starts_with_digit:
        tok_left = consume_left_numeric_designator(acr=acr, tokens=tokens, tok_left=tok_left)

    # 5) Build kept phrase inside the window: matched tokens + bridges + numeric-leading
    phrase = build_kept_phrase(
        tokens,
        tok_left=tok_left,
        tok_right=tok_right,
        hit_tokens=hit_tokens,
        bridges=bridges,
        include_numeric_leading=True,
    )
    if not phrase:
        return []

    # 6) Tight character spans over the ORIGINAL snippet
    base = m.start("pre")  # where `pre` starts in `snippet`
    ds = base + starts[tok_left]
    de = base + ends[tok_right]

    # 7) Normalize for display; indices stay tight
    norm = normalize_definition(phrase)
    if not norm:
        return []
    raw_window = collapse_ws(snippet[ds:de])  # raw chars between ds..de (just whitespace-collapsed)

    disp = normalize_definition(phrase)
    if not disp:
        return []
    return [LocalDefMatch(def_start=ds, def_end=de, definition=disp, raw=raw_window)]
