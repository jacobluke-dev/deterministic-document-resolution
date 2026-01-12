import re
from typing import Optional

from plainera_unacronym.nlp.common.constants_regex import DEFAULT_STOPWORDS, BRIDGES_DEFAULT
from plainera_unacronym.nlp.common.shared import normalize_definition
from plainera_unacronym.nlp.extraction.anchored.normalise import tighten_definition_span, strip_trailing_punct, collapse_ws

from plainera_unacronym.nlp.extraction.matchers.tighten import _initials_seq, _match_from, _split_compound


class LocalDefMatch:
    def __init__(self, def_start: int, def_end: int, definition: str, raw: str | None = None):
        self.def_start = def_start
        self.def_end = def_end
        self.definition = definition
        self.raw = raw


def has_letters(s: str) -> bool:
    """True if the string contains any Unicode letter.

    Args:
      s (str): String to check.

    Returns:
      bool: True if any character in ``s`` satisfies ``str.isalpha()``; else False.
    """
    return any(ch.isalpha() for ch in s)


# Hard clause boundary for inline defs (stop scanning / gating at these)
_INLINE_BOUNDARY_RE = re.compile(r"[.;:](?=\s|$)|[\r\n]")

def _inline_clause_tail(s: str) -> tuple[str, int]:
    """
    Return (tail_text, tail_end_index) where tail is from start of `s` up to
    the first hard boundary, or full `s` if none.
    """
    m = _INLINE_BOUNDARY_RE.search(s)
    end = m.start() if m else len(s)
    return s[:end], end


def _first_alnum_char_upper(s: str) -> str | None:
    for ch in s:
        if ch.isalnum():
            return ch.upper()
    return None


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

    if acr and require_initials_match:
        # Build initials / owners per *token*
        letters, owners = _initials_seq(tokens, getattr(cfg, "stopwords", DEFAULT_STOPWORDS))
        if not letters:
            return []

        # Build target acronym with per-char constraints
        A = [c for c in acr if c.isalnum()]
        if not A:
            return []

        # Forward scan to find the **shortest window** satisfying the constraints
        stop = getattr(cfg, "stopwords", DEFAULT_STOPWORDS)
        best = None  # (tok_s, tok_e, hit_token_indices)

        # Map token -> is_stopword once
        is_stop = [t.lower() in stop for t in tokens]

        # Per-letter constraint helper
        def ok_token_for(ch_upper: str, token_idx: int, matched_letter_pos: int) -> bool:
            # letters[matched_letter_pos] already equals ch_upper
            # enforce stopword vs non-stopword by case of the acronym letter
            return is_stop[token_idx] if A[matched_letter_pos].islower() else (not is_stop[token_idx])

        # We’ll reuse the existing `_match_from` over `letters`, then verify constraints
        L = [x.upper() for x in A]  # normalized targets for equality
        for li in range(len(letters)):
            r = _match_from(letters, L, li)
            if not r:
                continue
            lj, used_letter_pos = r  # lj = 1+last letter index in letters
            tok_s = owners[li]
            tok_e = owners[lj - 1]
            hits = {owners[u] for u in used_letter_pos}

            # enforce per-letter stopword constraint
            ok = True
            for k, letter_pos in enumerate(used_letter_pos):
                if not ok_token_for(L[k], owners[letter_pos], k):
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
    bridges = getattr(cfg, "bridges", BRIDGES_DEFAULT)
    kept = [t for idx, t in enumerate(tokens[i : j + 1]) if (i + idx) in hit_tokens or t.lower() in bridges]

    if not kept:  # edge case: keep original window
        kept = tokens[i : j + 1]

    phrase = " ".join(kept)
    phrase = strip_trailing_punct(collapse_ws(phrase))
    if not phrase:
        return []

    # 3a) Expand window to include adjacent numeric-leading tokens
    while i > 0:
        init = _first_alnum_char_upper(tokens[i - 1])
        if init is not None and not init.isalpha():  # e.g., "3M"
            i -= 1
        else:
            break

    while j + 1 < len(tokens):
        init = _first_alnum_char_upper(tokens[j + 1])
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
    bridges = getattr(cfg, "bridges", BRIDGES_DEFAULT)

    kept = []
    for idx in range(i, j + 1):
        tok = tokens[idx]
        init = _first_alnum_char_upper(tok)
        keep_numericish = (init is not None) and (not init.isalpha())
        if idx in hit_tokens or tok.lower() in bridges or keep_numericish:
            kept.append(tok)

    if not kept:
        kept = tokens[i : j + 1]

    phrase = strip_trailing_punct(collapse_ws(" ".join(kept)))
    if not phrase:
        return []

    # Use the window we computed (with numeric-leading expansion),
    # don't run tighten_definition_span on the whole raw.
    definition = normalize_definition(phrase)
    if not definition:
        return []

    return [LocalDefMatch(def_start=ds, def_end=de, definition=definition)]


def find_parenthetical_longform_before_acr(snippet: str, acr: str, cfg) -> list[LocalDefMatch]:
    """
    Find:  Long Form ... (ACR)   anchored at the end of `snippet`.

    Strategy (RIGHT→LEFT):
      1) Capture the text immediately before '(ACR)' with a look-ahead.
      2) Tokenize LTR (to get stable character spans), but *match* RTL.
      3) Split tokens into parts (hyphen/slash/dot/& and CamelCase).
         Build a per-part initials sequence RTL.
      4) Match ACR letters (ignoring non-alnum), with constraints:
         - UPPERCASE letter → must land on a non-stopword token
         - lowercase letter → must land on a stopword token
      5) The token window is [leftmost contributing token .. last token].
         Expand the window to include adjacent numeric-leading tokens.
      6) Keep matched tokens + bridges + numeric-leading tokens (for readability).
      7) Return tight character spans over the original `snippet` and a normalized phrase.
    """
    max_chars = getattr(cfg, "max_phrase_chars", 80)
    stop = getattr(cfg, "stopwords", DEFAULT_STOPWORDS)
    bridges = getattr(cfg, "bridges", BRIDGES_DEFAULT)

    # 0) Capture the preamble right before "(ACR)" anchored at end.
    acr_esc = re.escape(acr)
    m = re.search(
        rf"(?P<pre>[^\(\)]{{1,{max_chars}}})\s*(?=\(\s*{acr_esc}\s*\)\s*$)",
        snippet,
    )
    print("M IS ...", m)
    if not m:
        return []

    pre = m.group("pre").rstrip()
    if not pre or not has_letters(pre):
        return []

    # Raw length guard (before any tightening): enforce the configured limit strictly.
    if len(pre) > max_chars:
        return []

    # 1) Tokenize LTR by whitespace to get stable spans
    tokens: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for t in re.finditer(r"\S+", pre):
        tokens.append(t.group(0))
        starts.append(t.start())
        ends.append(t.end())

    if not tokens:
        return []

    is_stop = [tok.lower() in stop for tok in tokens]

    # 2) Build per-part initials RTL over tokens (compound + CamelCase aware)
    letters: list[str] = []  # per-part initials (UPPER)
    owners: list[int] = []  # token index for each letter
    part_is_stop: list[bool] = []  # stopword status of the owning token

    for ti in range(len(tokens) - 1, -1, -1):  # tokens RTL
        tok = tokens[ti]
        # split compounds (+ fallback CamelCase split if the _split_compound doesn't cover it)
        parts = _split_compound(tok)
        if not parts:  # very defensive
            continue
        for part in reversed(parts):  # parts RTL
            ch = _first_alnum_char_upper(part)
            if ch is None:
                continue
            letters.append(ch)
            owners.append(ti)
            part_is_stop.append(is_stop[ti])

    if not letters:
        return []

    # 3) Build target acronym letters (ignore non-alnum), match RTL
    A = [c for c in acr if c.isalnum()]
    if not A:
        return []

    j = len(A) - 1  # index in acronym (RTL)
    k = 0  # index in letters (already RTL order)
    used_letter_pos: list[int] = []

    while k < len(letters) and j >= 0:
        need = A[j].upper()
        want_stop = A[j].islower()  # lower-case letter forces stopword token
        if letters[k] == need and (part_is_stop[k] if want_stop else not part_is_stop[k]):
            used_letter_pos.append(k)
            j -= 1
        k += 1

    if j >= 0:
        return []  # failed to align all acronym letters

    # 4) The token window is from leftmost contributing token to the last token
    tok_right = len(tokens) - 1
    tok_left = min(owners[pos] for pos in used_letter_pos)
    hit_tokens = {owners[pos] for pos in used_letter_pos}

    # Expand window to include adjacent numeric-leading tokens (e.g., "3M")
    def _numeric_leading(idx: int) -> bool:
        init = _first_alnum_char_upper(tokens[idx])
        return (init is not None) and (not init.isalpha())

    while tok_left > 0 and _numeric_leading(tok_left - 1):
        tok_left -= 1
    while tok_right + 1 < len(tokens) and _numeric_leading(tok_right + 1):
        tok_right += 1

    # 5) Build kept phrase inside the window: matched tokens + bridges + numeric-leading
    kept_tokens: list[str] = []
    for idx in range(tok_left, tok_right + 1):
        tok = tokens[idx]
        keep_numericish = _numeric_leading(idx)
        if idx in hit_tokens or tok.lower() in bridges or keep_numericish:
            kept_tokens.append(tok)

    if not kept_tokens:  # extreme edge case: keep raw window
        kept_tokens = tokens[tok_left : tok_right + 1]

    phrase = " ".join(kept_tokens)
    phrase = strip_trailing_punct(collapse_ws(phrase))
    print("PHRASE:", phrase)
    if not phrase:
        return []

    # 6) Tight character spans over the ORIGINAL snippet
    base = m.start("pre")  # where `pre` starts in `snippet`
    ds = base + starts[tok_left]
    de = base + ends[tok_right]

    # 7) Normalize for display; indices stay tight
    norm = normalize_definition(phrase)
    if not norm:
        print("NORMAL:", norm)
        return []
    raw_window = collapse_ws(snippet[ds:de])  # raw chars between ds..de (just whitespace-collapsed)
    print("PHRASE:", phrase)
    disp = normalize_definition(tighten_definition_span(phrase))

    return [LocalDefMatch(def_start=ds, def_end=de, definition=disp, raw=raw_window)]

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

    # --- NEW: gate the whole inline clause tail (NOT the minimal initials window) ---
    tail, _ = _inline_clause_tail(snippet)

    print("TAIL_LEN:", len(collapse_ws(tail)), "MAX:", max_phrase_chars, "TAIL:", collapse_ws(tail)[:120])
    if len(collapse_ws(tail[0])) > max_phrase_chars:
        return []

    stop = getattr(cfg, "stop", DEFAULT_STOPWORDS)
    bridges = getattr(cfg, "bridges", BRIDGES_DEFAULT)
    max_phrase_chars = getattr(cfg, "max_phrase_chars", 200)
    search_cap = max_chars or max_phrase_chars * 2
    s = snippet[:search_cap]

    tail, _tail_end = _inline_clause_tail(s)
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
        phrase = strip_trailing_punct(collapse_ws(phrase))

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

    tokens, starts, ends = [], [], []
    for m in re.finditer(r"\S+", s):
        tokens.append(m.group(0)); starts.append(m.start()); ends.append(m.end())
    if not tokens:
        return []

    A_raw = [c for c in acr if c.isalnum()]
    if not A_raw:
        return []

    inits = [_first_alnum_char_upper(t) for t in tokens]
    is_stop = [t.lower() in stop for t in tokens]

    def ok_for_letter(letter: str, tok_idx: int) -> bool:
        return is_stop[tok_idx] if letter.islower() else (not is_stop[tok_idx])
    best = None
    L = [c.upper() for c in A_raw]  # matching uses uppercase equality

    # Greedy-forward scan for smallest window [i..j] that hits all letters in order
    for i in range(len(tokens)):
        li = 0
        hits: list[int] = []
        for j in range(i, len(tokens)):
            init = inits[j]
            if init == L[li] and ok_for_letter(A_raw[li], j):
                hits.append(j)
                li += 1
                if li == len(L):
                    # Found a window [i..j] with hit token indices = hits
                    if (best is None) or ((j - i) < (best[1] - best[0])):
                        best = (i, j, set(hits))
                    break  # try to shrink further by moving i forward
        # Early stop: if remaining tokens are fewer than remaining letters
        if len(tokens) - i < len(L):
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
    phrase = strip_trailing_punct(collapse_ws(phrase))

    # Respect max_phrase_chars after normalisation
    disp = normalize_definition(tighten_definition_span(phrase))
    if not disp or len(disp) > max_phrase_chars:
        return []

    return [LocalDefMatch(def_start=ds, def_end=de, definition=disp, raw=raw_window)]
