import re
from typing import Optional

from plainera_unacronym.nlp.common.constants_regex import BRIDGES_DEFAULT, DEFAULT_STOPWORDS
from plainera_unacronym.nlp.common.shared import (has_letters,
                                                  normalize_definition,
                                                  strip_trailing_punct_str,
                                                  collapse_ws)
from plainera_unacronym.nlp.extraction.anchored.normalise import tighten_definition_span
from plainera_unacronym.nlp.extraction.matchers.common import is_mixed_case_acronym
from plainera_unacronym.nlp.extraction.matchers.defs.common import (LocalDefMatch,
                                                                    first_alnum_char_upper,
                                                                    expand_numeric_leading_window,
                                                                    align_acronym_to_initials,
                                                                    build_initials_stream)


def find_parenthetical_longform_after_acr(
    snippet: str,
    cfg,
    acr: Optional[str] = None,
    require_initials_match: bool = True,  # renamed flag
) -> list[LocalDefMatch]:
    """Extract a parenthetical long form that appears *immediately after* an acronym.

    Parses ``snippet`` starting at the acronym's end and looks for a tight
    ``( … )`` that contains the expanded long form. If ``require_initials_match``
    is True and ``acr`` is provided, the function validates the long form by
    aligning the acronym letters (ignoring non-alnum) to the initials of the
    long-form tokens (compound- and CamelCase-aware), with these constraints:
    uppercase acronym letters must land on non-stopwords; lowercase letters
    must land on stopwords. The returned span tightly hugs the chosen token
    window, and the definition text is normalized for display.

    If ``require_initials_match`` is False, the raw inner text of the
    parentheses is used (after trimming outer spaces) without alignment.

    Args:
        snippet: Text that begins at, or right after, the acronym; the search
            is anchored at the start and expects ``( … )`` immediately after.
        cfg: Config object. Recognized attributes:
            - ``max_phrase_chars`` (int): Max characters allowed inside the
              parentheses (default: 80).
            - ``stopwords`` (set[str]): Tokens ignored for uppercase letters and
              required for lowercase letters during alignment.
            - ``bridges`` (set[str]): Extra tokens to keep inside the final
              window for readability (e.g., “of”, “for”).
        acr: The acronym to validate against (e.g., ``"PDF"``). Ignored when
            ``require_initials_match`` is False.
        require_initials_match: When True, only return a match if the acronym
            can be aligned to token initials as described above.

    Returns:
        A list with zero or one ``LocalDefMatch``:
        - ``def_start`` / ``def_end``: Tight character offsets into ``snippet``
          covering the chosen window inside the parentheses.
        - ``definition``: Normalized display string built from matched tokens
          plus any bridge and numeric-leading tokens within the window.

    Examples:
        >>> cfg = type("Cfg", (), {"max_phrase_chars": 80})()
        >>> find_parenthetical_longform_after_acr("(Portable Document Format)", cfg, acr="PDF")
        [LocalDefMatch(..., definition='Portable Document Format')]

        >>> # Disable alignment if you only want the parenthetical text
        >>> find_parenthetical_longform_after_acr("(noisy   RAW )", cfg, require_initials_match=False)
        [LocalDefMatch(..., definition='noisy RAW')]
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
    if not has_letters(raw_trim):
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
        kept = tokens[i: j + 1]

    phrase = strip_trailing_punct_str(collapse_ws(" ".join(kept)))
    if not phrase:
        return []

    definition = normalize_definition(phrase)
    if not definition:
        return []

    return [LocalDefMatch(def_start=ds, def_end=de, definition=definition)]
