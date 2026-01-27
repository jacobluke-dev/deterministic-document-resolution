import re

from plainera_unacronym.nlp.common.constants_regex import DEFAULT_STOPWORDS, BRIDGES_DEFAULT, QUOTE
from plainera_unacronym.nlp.common.shared import has_letters, strip_trailing_punct_str, collapse_ws, \
    normalize_definition
from plainera_unacronym.nlp.extraction.matchers.common import split_compound, is_mixed_case_acronym
from plainera_unacronym.nlp.extraction.matchers.defs.common import LocalDefMatch, is_acronym_like_token, \
    _acronym_letters_rtl, first_alnum_char_upper, has_numeric_evidence, acr_alignment_targets, align
from plainera_unacronym.nlp.extraction.matchers.numeric_matcher import consume_left_numeric_designator


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

    acr_esc = re.escape(acr)

    tail = r"(?:\s*[,;:]\s*[^)]{0,120})?"  # cap the tail to stay sane

    m = re.search(
        rf"(?P<pre>[^\(\)]{{1,{max_chars}}})\s*"
        rf"(?=\(\s*{QUOTE}{acr_esc}{QUOTE}{tail}\s*\)\s*$)",
        snippet,
    )

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

    acr_starts_with_digit = acr and acr[0].isdigit()

    # 2) Build per-part initials RTL over tokens (compound + CamelCase aware)
    letters: list[str] = []  # per-part initials (UPPER)
    owners: list[int] = []  # token index for each letter
    part_is_stop: list[bool] = []  # stopword status of the owning token

    for ti in range(len(tokens) - 1, -1, -1):  # tokens RTL
        tok = tokens[ti]

        # NEW: acronym-like token contributes multiple letters
        if is_acronym_like_token(tok):
            for ch in _acronym_letters_rtl(tok):
                letters.append(ch)
                owners.append(ti)
                part_is_stop.append(is_stop[ti])
            continue

        # Existing behaviour for normal words / compounds
        parts = split_compound(tok)
        if not parts:
            continue
        for part in reversed(parts):  # parts RTL
            ch = first_alnum_char_upper(part)
            if ch is None:
                continue
            letters.append(ch)
            owners.append(ti)
            part_is_stop.append(is_stop[ti])

    if not letters:
        return []

    # 3) Build target acronym letters (ignore non-alnum), match RTL
    has_num = has_numeric_evidence(tokens)
    A = acr_alignment_targets(acr, has_numeric_evidence=has_num)
    if not A:
        return []

    mixed = is_mixed_case_acronym(acr)

    used_letter_pos = align(
        A, letters, part_is_stop,
        allow_upper_on_stop=False,
        allow_lower_on_non_stop=mixed,
    )
    if used_letter_pos is None:
        used_letter_pos = align(
            A, letters, part_is_stop,
            allow_upper_on_stop=True,
            allow_lower_on_non_stop=mixed,
        )
    if used_letter_pos is None:
        return []

    # 4) The token window is from leftmost contributing token to the last token
    tok_right = len(tokens) - 1
    tok_left = min(owners[pos] for pos in used_letter_pos)
    if acr_starts_with_digit:
        tok_left = consume_left_numeric_designator(acr=acr, tokens=tokens, tok_left=tok_left)

    # --- include trailing numeric token for acronyms like HTTP2 -> "... 2" ---
    if acr and acr[-1].isdigit():
        want = acr[-1]

        # If last token is already numeric-leading with that digit, fine.
        # Otherwise, if there's an immediate next token equal to that digit, include it.
        if tok_right + 1 < len(tokens):
            nxt = tokens[tok_right + 1]
            nxt_clean = nxt.strip(".,;:)]}»”'\"")  # light trim
            if nxt_clean == want:
                tok_right += 1
    hit_tokens = {owners[pos] for pos in used_letter_pos}

    # Expand window to include adjacent numeric-leading tokens (e.g., "3M")
    def _numeric_leading(idx: int) -> bool:
        init = first_alnum_char_upper(tokens[idx])
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
    phrase = strip_trailing_punct_str(collapse_ws(phrase))
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
    print("PHRASE:", phrase)
    disp = normalize_definition(phrase)
    if not disp:
        return []
    return [LocalDefMatch(def_start=ds, def_end=de, definition=disp, raw=raw_window)]
