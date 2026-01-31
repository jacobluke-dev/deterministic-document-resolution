import re

from plainera_unacronym.nlp.common.constants_regex import DEFAULT_STOPWORDS, BRIDGES_DEFAULT
from plainera_unacronym.nlp.common.shared import collapse_ws, strip_trailing_punct_str, normalize_definition
from plainera_unacronym.nlp.extraction.anchored.normalise import tighten_definition_span
from plainera_unacronym.nlp.extraction.matchers.common import is_mixed_case_acronym
from plainera_unacronym.nlp.extraction.matchers.defs.common import (LocalDefMatch,
                                                                    inline_clause_tail,
                                                                    strip_inline_cue_prefix,
                                                                    kept_token_indices,
                                                                    build_initials_stream,
                                                                    align_acronym_to_initials,
                                                                    phrase_from_indices)

def find_inline_longform_after_acr(
    snippet: str,
    cfg,
    acr: str,
    *,
    max_chars: int | None = None,
    require_initials_match: bool = True,
) -> list[LocalDefMatch]:
    """
    Find an inline long-form definition that appears immediately after an acronym.

    This matcher targets patterns where an acronym is followed by cue words and then a
    definitional phrase, e.g.:

        "PDF stands for Portable Document Format"
        "API means Application Programming Interface"
        "NLP (in this context) refers to Natural Language Processing"

    The function returns at most one `LocalDefMatch` containing:
      - `def_start`/`def_end`: character offsets into the *input slice* being searched
      - `definition`: a normalized display string for the extracted long-form
      - `raw`: the raw extracted window with whitespace collapsed (for debugging/audit)

    The matcher has two modes:

    1) Initials-matched mode (`require_initials_match=True`, default)
       - Uses `strip_inline_cue_prefix()` to require and remove leading cue words
         (e.g., "stands for", "means", "is short for", etc.).
       - Tokenizes the remaining tail and aligns `acr` to the token initials stream
         using `align_acronym_to_initials()` (LTR minimal window).
       - Expands the kept tokens using `kept_token_indices()` to include bridges
         (e.g., "of") and numeric-leading tokens for readability.
       - Produces a normalized definition via `tighten_definition_span()` and
         `normalize_definition()`.

    2) Fast-path mode (`require_initials_match=False`)
       - Does not require cue words or initials alignment.
       - Takes up to 6 tokens from the beginning of the searchable slice, or stops
         earlier on a hard clause boundary token ('.', ':', ';').
       - Normalizes and returns that bounded phrase (subject to length gates).

    Length gating:
      - The function first gates the full inline clause tail (up to the first hard
        boundary, via `inline_clause_tail()`) against `cfg.max_phrase_chars`.
      - It then searches within a bounded prefix of `snippet` (`search_cap`), where:
            search_cap = max_chars if provided else cfg.max_phrase_chars * 2
      - Both the raw extracted window and the normalized display definition are
        rejected if they exceed `cfg.max_phrase_chars`.

    Configuration:
      - `cfg.max_phrase_chars` (int, default 200): maximum allowed length for the
        extracted definition and raw window.
      - `cfg.stop` (set[str], optional): stopword set used during initials alignment
        (defaults to `DEFAULT_STOPWORDS` if missing).
      - `cfg.bridges` (set[str], optional): tokens permitted between matched tokens
        when building the kept phrase (defaults to `BRIDGES_DEFAULT` if missing).

    Args:
        snippet: Text beginning at/near the inline definition clause to be scanned.
        cfg: Configuration object providing limits and optional token sets.
        acr: The acronym to resolve (e.g., "PDF").
        max_chars: Optional cap on how many characters of `snippet` are searched.
        require_initials_match: If True, require cue words + initials alignment.
            If False, use a short heuristic extraction without alignment.

    Returns:
        A list containing zero or one `LocalDefMatch`. Returns an empty list if no
        suitable inline definition is found or if any length/normalization gate fails.
    """
    if not snippet:
        return []
    max_phrase_chars = getattr(cfg, "max_phrase_chars", 200)

    # --- gate the whole inline clause tail (NOT the minimal initials window) ---
    tail, _ = inline_clause_tail(snippet)

    if len(collapse_ws(tail)) > max_phrase_chars:
        return []

    stop = getattr(cfg, "stop", DEFAULT_STOPWORDS)
    bridges = getattr(cfg, "bridges", BRIDGES_DEFAULT)
    max_phrase_chars = getattr(cfg, "max_phrase_chars", 200)
    search_cap = max_chars or max_phrase_chars * 2
    s = snippet[:search_cap]

    tail, _tail_end = inline_clause_tail(s)
    tail_collapsed = collapse_ws(tail)
    if len(tail_collapsed) > max_phrase_chars:
        return []

    # Fast path: if initials matching is NOT required, take a short bounded phrase.
    if not require_initials_match:
        # take up to N tokens (e.g., 6) or stop at a hard clause boundary
        tokens, starts, ends = [], [], []
        for m in re.finditer(r"\S+", s):
            tok = m.group(0)
            tokens.append(tok)
            starts.append(m.start())
            ends.append(m.end())
            # bail early if we hit a clear clause boundary token at the end
            if tok.endswith((".", ":", ";")):
                break
            if len(tokens) >= 6:
                break
        if not tokens:
            return []
        ds, de = starts[0], ends[-1]
        phrase = " ".join(tokens)
        phrase = strip_trailing_punct_str(collapse_ws(phrase))

        raw_window = collapse_ws(s[ds:de])  # raw chars between ds..de (just whitespace-collapsed)
        if len(raw_window) > max_phrase_chars:  # <-- gate HERE
            return []
        disp = normalize_definition(tighten_definition_span(phrase))
        if len(disp) > max_phrase_chars:
            return []
        return [LocalDefMatch(def_start=ds, def_end=de, definition=disp, raw=raw_window)]

    # Only align against the long-form tail, not the cue words.
    hit = strip_inline_cue_prefix(s, cfg)
    if hit:
        tail2, off = hit
    else:
        # If we cannot see a cue at the start, this isn't the pattern we're targeting.
        return []

    tokens, starts, ends = [], [], []
    for m in re.finditer(r"\S+", tail2):
        tokens.append(m.group(0))
        starts.append(m.start() + off)  # shift spans back into `s`
        ends.append(m.end() + off)

    if not tokens:
        return []

    stream = build_initials_stream(
        tokens,
        stopwords=stop,
        scan="ltr",
        expand_allcaps_tokens=is_mixed_case_acronym(acr),
        split_compounds=False,
        treat_acronym_tokens_as_multi_letter=False,
    )

    hit = align_acronym_to_initials(
        acr,
        stream,
        tokens=tokens,
        stopwords=stop,
        mode="ltr_min_window",
        allow_upper_on_stop=False,
        allow_lower_on_non_stop=is_mixed_case_acronym(acr),
        lowercase_prefix_exception=True,
    )
    if hit is None:
        hit = align_acronym_to_initials(
            acr,
            stream,
            tokens=tokens,
            stopwords=stop,
            mode="ltr_min_window",
            allow_upper_on_stop=True,
            allow_lower_on_non_stop=is_mixed_case_acronym(acr),
            lowercase_prefix_exception=True,
        )
    if hit is None:
        return []

    i, j = hit.tok_left, hit.tok_right
    hit_tokens = hit.hit_tokens

    kept_idx = kept_token_indices(
        tokens,
        tok_left=i,
        tok_right=j,
        hit_tokens=hit_tokens,
        bridges=bridges,
        include_numeric_leading=True
    )

    if not kept_idx:
        return []

    ds, de = starts[kept_idx[0]], ends[kept_idx[-1]]

    raw_window = collapse_ws(s[ds:de])
    if len(raw_window) > max_phrase_chars:
        return []

    phrase = phrase_from_indices(tokens, kept_idx)
    disp = normalize_definition(tighten_definition_span(phrase))
    if not disp or len(disp) > max_phrase_chars:
        return []

    return [LocalDefMatch(def_start=ds, def_end=de, definition=disp, raw=raw_window)]
