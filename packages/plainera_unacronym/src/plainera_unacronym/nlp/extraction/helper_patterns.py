import re
from typing import Optional

from .tighten import _initials_seq, _match_from, _split_compound
from ..common.constants import DEFAULT_STOPWORDS, BRIDGES_DEFAULT
from ..common.shared import tighten_definition_span, normalize_definition, strip_trailing_punct, \
    collapse_ws


class LocalDefMatch:
    def __init__(self, def_start: int, def_end: int, definition: str):
        self.def_start = def_start
        self.def_end = def_end
        self.definition = definition


def _has_letters(s: str) -> bool:
    """True if the string contains any Unicode letter.

    Args:
      s (str): String to check.

    Returns:
      bool: True if any character in ``s`` satisfies ``str.isalpha()``; else False.
    """
    return any(ch.isalpha() for ch in s)


def _initials_match(acr: str, phrase: str) -> bool:
    """Check if an acronym fits the phrase's initials as an ordered subsequence.

        Builds an uppercase string of initials from the phrase by taking the first
        character of each word **only if** that character is alphabetic. Then checks
        whether the alphabetic characters of ``acr`` (ignoring any non-letters in
        ``acr``) appear in order within those initials.

        This is case-insensitive for matching and does not require contiguity—only
        order. Words that begin with non-letters (e.g., ``"3M"``, ``"7-Document"``)
        do not contribute an initial.

        Args:
          acr (str): The Acronym to test.
          phrase (str): Candidate long-form phrase used to derive initials.

        Returns:
          bool: True if the acronym's letters appear in order within the phrase initials;
          otherwise False.

        """
    initials = "".join(w[0].upper() for w in phrase.split() if w and w[0].isalpha())
    j = 0
    for ch in acr:
        if ch.isalpha():
            j = initials.find(ch, j) + 1
            if j == 0:
                return False
    return True

def _first_alnum_char_upper(s: str) -> str | None:
    for ch in s:
        if ch.isalnum():
            return ch.upper()
    return None

def find_longform_after_acr(
    snippet: str,
    cfg,
    acr: Optional[str] = None,
    require_initials_match: bool = True,  # renamed flag
) -> list[LocalDefMatch]:
    max_chars = getattr(cfg, "max_phrase_chars", 80)

    # 1) Capture tight inner text "( ... )" right at the start of `snippet`
    m = re.match(rf"\A\s*\((?P<def>[^()]{{1,{max_chars}}}?)(?=\s*\))\s*\)", snippet)
    if not m:
        return []

    raw = m.group("def")
    raw_trim = raw.strip()
    if not _has_letters(raw_trim):
        return []
    if len(raw_trim) > max_chars:
        return []  # raw length guard (prevents tail slices)

    # 2) Tokenize the raw parenthetical (preserve case; no normalization yet)
    tokens = raw_trim.split()  # or your _tokenize_preserve(raw_trim)
    if not tokens:
        return []

    if acr and require_initials_match:
        # Build initials / owners per *token* (or per-part if that’s your global choice)
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

        # We’ll reuse your existing `_match_from` over `letters`, then verify constraints
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
    kept = [t for idx, t in enumerate(tokens[i:j + 1])
            if (i + idx) in hit_tokens or t.lower() in bridges]

    if not kept:  # edge case: keep original window
        kept = tokens[i:j + 1]

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

    # 4) Tight character spans over the original `snippet`
    left_str = " ".join(tokens[:i])
    keep_str = " ".join(tokens[i:j + 1])
    left_offset = (len(left_str) + 1) if left_str else 0
    ds = m.start("def") + left_offset
    de = ds + len(keep_str)

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
        kept = tokens[i:j + 1]

    phrase = strip_trailing_punct(collapse_ws(" ".join(kept)))
    if not phrase:
        return []

    norm = normalize_definition(phrase)
    if not norm:
        return []

    return [LocalDefMatch(def_start=ds, def_end=de, definition=norm)]


def _first_alnum(s: str) -> str | None:
    for ch in s:
        if ch.isalnum():
            return ch.upper()
    return None

def find_longform_before_acr(snippet: str, acr: str, cfg) -> list[LocalDefMatch]:
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
    if not m:
        return []

    pre = m.group("pre").rstrip()
    if not pre or not _has_letters(pre):
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
    letters: list[str] = []       # per-part initials (UPPER)
    owners: list[int] = []        # token index for each letter
    part_is_stop: list[bool] = [] # stopword status of the owning token

    for ti in range(len(tokens) - 1, -1, -1):  # tokens RTL
        tok = tokens[ti]
        # split compounds (+ fallback CamelCase split if your _split_compound doesn't cover it)
        parts = _split_compound(tok)
        if not parts:  # very defensive
            continue
        for part in reversed(parts):           # parts RTL
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

    j = len(A) - 1                # index in acronym (RTL)
    k = 0                         # index in letters (already RTL order)
    used_letter_pos: list[int] = []

    while k < len(letters) and j >= 0:
        need = A[j].upper()
        want_stop = A[j].islower()   # lower-case letter forces stopword token
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
        kept_tokens = tokens[tok_left:tok_right + 1]

    phrase = " ".join(kept_tokens)
    phrase = strip_trailing_punct(collapse_ws(phrase))
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

    return [LocalDefMatch(def_start=ds, def_end=de, definition=norm)]
