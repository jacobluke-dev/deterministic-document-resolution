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
                                                                    phrase_from_indices,
                                                                    first_alnum_char_upper)


def find_inline_longform_after_acr(
    snippet: str,
    cfg,
    acr: str,
    *,
    max_chars: int | None = None,
    require_initials_match: bool = True,
) -> list[LocalDefMatch]:
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
        include_numeric_leading=True,
        first_alnum_char_upper=first_alnum_char_upper,
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
