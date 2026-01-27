import re

from plainera_unacronym.nlp.common.constants_regex import DEFAULT_STOPWORDS, BRIDGES_DEFAULT
from plainera_unacronym.nlp.common.shared import collapse_ws, strip_trailing_punct_str, normalize_definition
from plainera_unacronym.nlp.extraction.anchored.normalise import tighten_definition_span
from plainera_unacronym.nlp.extraction.matchers.common import is_mixed_case_acronym, PUNCT_TRIM
from plainera_unacronym.nlp.extraction.matchers.defs.common import LocalDefMatch, inline_clause_tail, \
    strip_inline_cue_prefix, has_numeric_evidence, acr_alignment_targets


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

    if len(collapse_ws(tail[0])) > max_phrase_chars:
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
            tokens.append(tok); starts.append(m.start()); ends.append(m.end())
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

    # ---- initials-matching path (as before) ----
    def _first_alnum_char_upper(tok: str) -> str | None:
        for ch in tok:
            if ch.isalnum():
                return ch.upper()
        return None

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

    has_num = has_numeric_evidence(tokens)
    A_raw = acr_alignment_targets(acr, has_numeric_evidence=has_num)
    if not A_raw:
        return []

    letters: list[str] = []
    owners: list[int] = []  # letter index → token index

    mixed = is_mixed_case_acronym(acr)

    for ti, tok in enumerate(tokens):
        tok_clean = tok.strip(PUNCT_TRIM)

        # Expand ALLCAPS tokens only for mixed-case acronyms
        if mixed and tok_clean.isalpha() and tok_clean.isupper() and len(tok_clean) > 1:
            for ch in tok_clean:
                letters.append(ch.upper())
                owners.append(ti)
            continue

        ch = _first_alnum_char_upper(tok_clean)
        if ch:
            letters.append(ch)
            owners.append(ti)

    if not letters:
        return []

    is_stop = [t.lower() in stop for t in tokens]

    def ok_for_letter(letter: str, tok_idx: int, acr_pos: int) -> bool:
        # uppercase -> non-stopword
        if not letter.islower():
            return not is_stop[tok_idx]

        # lowercase -> stopword (default)
        if is_stop[tok_idx]:
            return True

        # ---- narrow exception: mixed-case leading lowercase (mRNA / iOS) ----
        if not mixed:
            return False
        if acr_pos != 0:
            return False
        if tok_idx != 0:
            return False

        tok0 = tokens[0].strip(PUNCT_TRIM)
        if tok0.isalpha() and tok0.isupper():  # don't let acronym-like token satisfy lowercase prefix
            return False

        return tok0[:1].lower() == letter.lower()

    # Greedy-forward scan for smallest window [i..j] that hits all letters in order
    best = None
    L = [c.upper() for c in A_raw]

    for li in range(len(letters)):
        ai = 0
        hit_letters: list[int] = []

        for lj in range(li, len(letters)):
            if letters[lj] == L[ai] and ok_for_letter(A_raw[ai], owners[lj], ai):
                hit_letters.append(lj)
                ai += 1
                if ai == len(L):
                    tok_s = owners[li]
                    tok_e = owners[lj]
                    hit_tokens = {owners[h] for h in hit_letters}

                    if best is None or (tok_e - tok_s) < (best[1] - best[0]):
                        best = (tok_s, tok_e, hit_tokens)
                    break

        if len(letters) - li < len(L):
            break

    if not best:
        return []

    i, j, hit_tokens = best
    kept_idx = [idx for idx in range(i, j + 1) if idx in hit_tokens or tokens[idx].lower() in bridges] or list(range(i, j + 1))
    ds, de = starts[kept_idx[0]], ends[kept_idx[-1]]

    # reject if the raw candidate span is too long (don’t truncate)
    raw_window = collapse_ws(s[ds:de])
    if len(raw_window) > max_phrase_chars:
        return []

    phrase = " ".join(tokens[k] for k in kept_idx)
    phrase = strip_trailing_punct_str(collapse_ws(phrase))

    # Respect max_phrase_chars after normalisation
    disp = normalize_definition(tighten_definition_span(phrase))
    if not disp or len(disp) > max_phrase_chars:
        return []

    return [LocalDefMatch(def_start=ds, def_end=de, definition=disp, raw=raw_window)]
