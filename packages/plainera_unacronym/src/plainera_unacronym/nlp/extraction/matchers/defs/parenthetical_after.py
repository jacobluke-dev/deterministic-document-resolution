import re
from typing import Optional

from plainera_unacronym.nlp.common.constants_regex import BRIDGES_DEFAULT, DEFAULT_STOPWORDS
from plainera_unacronym.nlp.common.shared import has_letters, normalize_definition, strip_trailing_punct_str, \
    collapse_ws
from plainera_unacronym.nlp.extraction.anchored.normalise import tighten_definition_span
from plainera_unacronym.nlp.extraction.matchers.common import is_mixed_case_acronym, initials_seq, match_from
from plainera_unacronym.nlp.extraction.matchers.defs.common import LocalDefMatch, has_numeric_evidence, \
    acr_alignment_targets, first_alnum_char_upper


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
    mixed = acr and is_mixed_case_acronym(acr)

    if not tokens:
        return []

    if acr and require_initials_match:
        # Build initials / owners per *token*
        letters, owners = initials_seq(tokens, expand_allcaps=is_mixed_case_acronym(acr))
        if not letters:
            return []

        # Build target acronym with per-char constraints
        has_num = has_numeric_evidence(tokens)
        A = acr_alignment_targets(acr, has_numeric_evidence=has_num)
        if not A:
            return []

        # Forward scan to find the **shortest window** satisfying the constraints
        best = None  # (tok_s, tok_e, hit_token_indices)

        # Map token -> is_stopword once
        is_stop = [t.lower() in stop for t in tokens]

        # Per-letter constraint helper
        mixed = bool(acr) and is_mixed_case_acronym(acr)

        def ok_token_for(token_idx: int, acr_pos: int) -> bool:
            """
            Enforce stopword/non-stopword constraints by acronym letter case.

            Default contract (strict):
              - uppercase acronym letter -> must land on NON-stopword
              - lowercase acronym letter -> must land on stopword

            Narrow exception (to support mRNA/iOS-style prefixes without breaking other tests):
              - allow a lowercase *first* acronym letter to land on a non-stopword token
                ONLY if it maps to the first token and that token starts with the same letter.
            """
            want_stop = A[acr_pos].islower()
            if not want_stop:
                return not is_stop[token_idx]

            # strict default: lowercase must land on stopword
            if is_stop[token_idx]:
                return True

            # ---- narrow exception ----
            if not mixed:
                return False
            if acr_pos != 0:
                return False
            if token_idx != 0:
                return False

            tok0 = tokens[0]
            # Don't allow acronym tokens (ALLCAPS) to satisfy lowercase prefixes
            if tok0.isalpha() and tok0.isupper():
                return False

            return tok0[:1].lower() == A[0].lower()

        # We’ll reuse the existing `_match_from` over `letters`, then verify constraints
        L = [x.upper() for x in A]  # normalized targets for equality
        for li in range(len(letters)):
            r = match_from(letters, L, li)
            if not r:
                continue
            lj, used_letter_pos = r  # lj = 1+last letter index in letters
            tok_s = owners[li]
            tok_e = owners[lj - 1]
            hits = {owners[u] for u in used_letter_pos}

            # enforce per-letter stopword constraint
            ok = True
            for k, letter_pos in enumerate(used_letter_pos):
                if not ok_token_for(owners[letter_pos], k):
                    ok = False
                    break
            if not ok:
                continue

            if best is None or (tok_e - tok_s) < (best[1] - best[0]):
                best = (tok_s, tok_e, hits)

        if not best:
            return []

        i, j, hit_tokens = best
    else:
        # If matching is disabled, keep the whole raw_trim as a single-window
        i, j = 0, len(tokens) - 1
        hit_tokens = set(range(i, j + 1))

    # 3) Build kept phrase: matched tokens + bridges inside the window
    kept = [t for idx, t in enumerate(tokens[i : j + 1]) if (i + idx) in hit_tokens or t.lower() in bridges]

    if not kept:  # edge case: keep original window
        kept = tokens[i : j + 1]

    phrase = " ".join(kept)
    phrase = strip_trailing_punct_str(collapse_ws(phrase))
    if not phrase:
        return []

    # 3a) Expand window to include adjacent numeric-leading tokens
    while i > 0:
        init = first_alnum_char_upper(tokens[i - 1])
        if init is not None and not init.isalpha():  # e.g., "3M"
            i -= 1
        else:
            break

    while j + 1 < len(tokens):
        init = first_alnum_char_upper(tokens[j + 1])
        if init is not None and not init.isalpha():
            j += 1
        else:
            break

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

    # 5) Build kept phrase: matched tokens + bridges + numeric-leading tokens inside the window

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

    # Use the window we computed (with numeric-leading expansion),
    # don't run tighten_definition_span on the whole raw.
    definition = normalize_definition(phrase)
    if not definition:
        return []

    return [LocalDefMatch(def_start=ds, def_end=de, definition=definition)]
