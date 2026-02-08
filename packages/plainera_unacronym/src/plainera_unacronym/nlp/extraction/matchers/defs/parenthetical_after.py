import re
from typing import Optional

from plainera_unacronym.nlp.common.constants_regex import BRIDGES_DEFAULT, DEFAULT_STOPWORDS
from plainera_unacronym.nlp.common.shared import collapse_ws, has_letter, strip_trailing_punct_str
from plainera_unacronym.nlp.extraction.anchored.normalise import tighten_definition_span
from plainera_unacronym.nlp.extraction.core.normalise import normalize_definition
from plainera_unacronym.nlp.extraction.matchers.common import is_mixed_case_acronym
from plainera_unacronym.nlp.extraction.matchers.defs.common import (
    LocalDefMatch,
    align_acronym_to_initials,
    build_initials_stream,
    expand_numeric_leading_window,
    first_alnum_char_upper,
)


def find_parenthetical_longform_after_acr(  # noqa: C901
    snippet: str,
    cfg,
    acr: Optional[str] = None,
    require_initials_match: bool = True,  # renamed flag
) -> list[LocalDefMatch]:
    """Extract a parenthetical long-form from the start of `snippet`.

    The function is *anchored at the beginning* of `snippet` and looks for a
    tight parenthetical of the form ``( ... )`` (allowing leading whitespace).
    It captures the inner text up to `cfg.max_phrase_chars` characters, rejects
    non-letter content, and returns at most one `LocalDefMatch`.

    If `require_initials_match` is False, the trimmed inner parenthetical text
    is passed through `tighten_definition_span()` then `normalize_definition()`
    and returned as-is. The returned offsets (`def_start`, `def_end`) tightly
    hug the inner text (excluding inner leading/trailing spaces).

    If `require_initials_match` is True and `acr` is provided (truthy), the
    inner text is tokenised (whitespace split), an initials stream is built via
    `build_initials_stream()`, and the acronym is aligned to token initials via
    `align_acronym_to_initials()` in `ltr_min_window` mode. Alignment is tried
    first with stricter stopword constraints, then re-tried with a relaxed
    `allow_upper_on_stop=True` fallback. The matched token window may be
    expanded leftward by `expand_numeric_leading_window()`. The display
    definition is constructed from hit tokens plus any bridge tokens
    (`cfg.bridges`) and numeric-leading tokens inside the final window, then
    normalised for display.

    Note: if `require_initials_match` is True but `acr` is missing/empty, no
    alignment is performed and the full parenthetical content is returned
    (after normalisation), since there is nothing to validate against.

    Args:
        snippet: Text expected to start with optional whitespace followed by
            a parenthetical long-form, e.g. ``"(Portable Document Format) ..."``.
        cfg: Config object. Recognised attributes:
            - max_phrase_chars (int): Maximum characters inside parentheses (default 80)
            - stop (set[str]) or stopwords (set[str]): stopword set for alignment
            - bridges (set[str]): tokens kept for readability when building the display definition
        acr: Acronym to validate against when `require_initials_match` is True.
        require_initials_match: When True and `acr` is provided, only return a
            match if acronym-to-initials alignment succeeds.

    Returns:
        list[LocalDefMatch]: zero or one match with offsets into `snippet`
        and a normalised definition string.
    """

    bridges = getattr(cfg, "bridges", BRIDGES_DEFAULT)
    stop = getattr(cfg, "stop", None) or getattr(cfg, "stopwords", DEFAULT_STOPWORDS)
    max_chars = getattr(cfg, "max_phrase_chars", 80)

    # 1) Capture tight inner text "( ... )" right at the start of `snippet`
    m = re.match(rf"\A\s*\((?P<def>[^()]{{1,{max_chars}}}?)(?=\s*\))\s*\)", snippet)
    if not m:
        return []

    raw = m.group("def")
    raw_trim = raw.strip()
    if not has_letter(raw_trim):
        return []
    if len(raw_trim) > max_chars:
        return []

    if not require_initials_match:
        # Tight span over the ORIGINAL inner text (preserve inner spaces)
        lead_ws = len(raw) - len(raw.lstrip())
        trail_ws = len(raw) - len(raw.rstrip())
        ds = m.start("def") + lead_ws
        de = m.end("def") - trail_ws

        # Run the display pipeline on the trimmed inner text
        tightened = tighten_definition_span(raw_trim)
        definition = normalize_definition(tightened)
        if not definition:
            return []
        return [LocalDefMatch(def_start=ds, def_end=de, definition=definition)]

    # 2) Tokenize the raw parenthetical (preserve case; no normalization yet)
    tokens = raw_trim.split()
    if not tokens:
        return []

    # ---- choose window (i, j) and hit_tokens ----
    if acr and require_initials_match:
        needs_compound_split = any(("-" in t) or ("/" in t) or ("&" in t) or ("." in t) for t in tokens)

        stream = build_initials_stream(
            tokens,
            stopwords=stop,
            scan="ltr",
            expand_allcaps_tokens=is_mixed_case_acronym(acr),
            split_compounds=needs_compound_split,
            treat_acronym_tokens_as_multi_letter=needs_compound_split,
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
    else:
        i, j = 0, len(tokens) - 1
        hit_tokens = set(range(i, j + 1))

    # ---- numeric-leading expansion (single place) ----
    i, j = expand_numeric_leading_window(tokens, i, j)

    # ---- compute token spans within raw_trim ----
    spans = []
    cursor = 0
    for t in tokens:
        pos = raw_trim.find(t, cursor)
        spans.append((pos, pos + len(t)))
        cursor = pos + len(t)

    # 4) Tight character spans over the original `snippet`
    lead_ws = len(raw) - len(raw.lstrip())
    raw_def_start = m.start("def") + lead_ws
    tok_start, _ = spans[i]
    _, tok_end = spans[j]
    ds = raw_def_start + tok_start
    de = raw_def_start + tok_end

    # ---- build kept tokens (hits + bridges + numeric-leading tokens) ----
    kept = []
    for idx in range(i, j + 1):
        tok = tokens[idx]
        init = first_alnum_char_upper(tok)
        keep_numericish = (init is not None) and (not init.isalpha())
        if idx in hit_tokens or tok.lower() in bridges or keep_numericish:
            kept.append(tok)

    if not kept:
        kept = tokens[i : j + 1]

    phrase = strip_trailing_punct_str(collapse_ws(" ".join(kept)))
    if not phrase:
        return []

    definition = normalize_definition(phrase)
    if not definition:
        return []

    return [LocalDefMatch(def_start=ds, def_end=de, definition=definition)]
