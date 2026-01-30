import re
from typing import Optional

from plainera_unacronym.nlp.common.constants_regex import BRIDGES_DEFAULT
from plainera_unacronym.nlp.common.shared import canonicalize, strip_trailing_punct_str, collapse_ws
from plainera_unacronym.nlp.extraction.matchers.common import match_from, initials_seq, is_mixed_case_acronym

_word_re = re.compile(r"[A-Za-z0-9'’\-\/&\.]+", flags=re.UNICODE)


def _tokenize_preserve(text: str) -> list[str]:
    """Tokenise `text` into “word-ish” tokens while preserving certain punctuation.

        This tokenizer is intentionally conservative and ASCII-centric: it extracts runs
        of characters matching `[A-Za-z0-9'’\\-\\/&\\.]+` and returns them as tokens.
        It preserves internal punctuation that commonly appears in acronyms, names, and
        technical identifiers (e.g. hyphens, slashes, ampersands, dots, apostrophes).

        Examples:
            - "Foo-Bar"      -> ["Foo-Bar"]
            - "Foo/Bar"      -> ["Foo/Bar"]
            - "R&D"          -> ["R&D"]
            - "U.S.A."       -> ["U.S.A."]
            - "can't / don’t"-> ["can't", "don’t"]

        Notes:
            - Non-ASCII letters act as boundaries; ASCII runs around them can still
              be emitted as separate tokens.
            - Whitespace and other punctuation (e.g. commas, parentheses) act as
              boundaries and are not included.

        Args:
            text (str): Input string to tokenise.

        Returns:
            list[str]: List of matched tokens (possibly empty).
        """
    return _word_re.findall(text)


def _best_window_for_acronym(
    tokens: list[str], acronym: str
) -> Optional[tuple[int, int, set[int]]]:
    """
    Return (tok_start, tok_end_inclusive, hit_token_indices_set) for the *shortest contiguous*
    token window whose (non-stopword, compound-aware) initials match `acronym` in order.
    """
    A = [c.upper() for c in acronym if c.isalnum()]
    if not A:
        return None

    letters, owners = initials_seq(tokens, expand_allcaps=is_mixed_case_acronym(acronym))
    if not letters:
        return None

    best = None  # (tok_s, tok_e, hits_set)
    for li in range(len(letters)):
        res = match_from(letters, A, li)
        if not res:
            continue
        lj, used_letters = res
        tok_s = owners[li]
        tok_e = owners[lj - 1]
        # collect which tokens actually contributed matched initials
        hits = {owners[u] for u in used_letters}
        if best is None or (tok_e - tok_s) < (best[1] - best[0]):
            best = (tok_s, tok_e, hits)
    return best


_NUMERIC_LEADING_RE = re.compile(r"^[^A-Za-z]*[0-9]")  # fast-ish heuristic


def _numeric_leading(tok: str) -> bool:
    """Return True if `tok` begins with a digit once leading punctuation is ignored.

    Scans LTR for the first alphanumeric character. Returns True if that
    character is a digit, otherwise False. If `tok` contains no alphanumeric
    characters, returns False.

    Examples:
        - "3M" -> True
        - "5th" -> True
        - "(12V)" -> True
        - "GPU" -> False
        - "--" -> False

    Args:
        tok (str): Token to inspect (may include punctuation).

    Returns:
        bool: True if the first alphanumeric character is a digit; otherwise False.
    """
    for ch in tok:
        if ch.isalnum():
            return not ch.isalpha()
    return False


_SPLIT_ACR_MARKER_RE = re.compile(r"[&./-]")


def _try_split_acronym_initials_window(
    *,
    tokens: list[str],
    acronym: str,
    bridges: set[str],
    keep_case: bool,
) -> str | None:
    """Extract a minimal initials-aligned token window for split acronyms.

    This helper targets acronyms that *explicitly* contain split markers (e.g. `/`, `&`, `.`, `-`)
    such as ``"C/A"``, ``"R&D"``, or ``"A/B/C"``. It attempts to find a subsequence of token
    initials that matches the acronym's alphanumeric characters in order, then returns a pruned
    phrase consisting of:

    - Tokens whose initials participated in the match,
    - Any bridge tokens (e.g. "and", "of") within the matched window,
    - Any numeric-leading tokens within the window (e.g. "3M", "5th") and numeric-leading
      neighbours immediately adjacent to the window.

    If no full initials sequence is found, returns ``None`` so the caller can fall back to the
    default tightening strategy.

    Args:
        tokens (list[str]): Tokenised phrase (case-preserving).
        acronym (str): Acronym surface form, used to derive the initials sequence.
        bridges (set[str]): Lowercased bridge words to retain when inside the matched window.
        keep_case (bool): If True, preserve original casing in the returned phrase; otherwise
            return a lowercased phrase.

    Returns:
        str | None: The tightened phrase if a match is found, otherwise ``None``.
    """
    if not _SPLIT_ACR_MARKER_RE.search(acronym):
        return None

    letters = [c for c in re.sub(r"[^A-Za-z0-9]+", "", acronym).upper()]
    if len(letters) < 2:
        return None

    # find first matching subsequence by token initial
    seq_idxs: list[int] = []
    li = 0
    for idx, tok in enumerate(tokens):
        ch = tok[0].upper() if tok else ""
        if ch == letters[li]:
            seq_idxs.append(idx)
            li += 1
            if li == len(letters):
                break

    if len(seq_idxs) != len(letters):
        return None

    low, high = min(seq_idxs), max(seq_idxs)

    # expand around numeric-leading neighbours (3M / 5th / etc.)
    while low > 0 and _numeric_leading(tokens[low - 1]):
        low -= 1
    while high + 1 < len(tokens) and _numeric_leading(tokens[high + 1]):
        high += 1

    hit_set = set(seq_idxs)
    kept: list[str] = []
    for idx in range(low, high + 1):
        tok = tokens[idx]
        if idx in hit_set or tok.lower() in bridges or _numeric_leading(tok):
            kept.append(tok)

    if not kept:
        kept = tokens[low: high + 1]

    phrase = strip_trailing_punct_str(collapse_ws(" ".join(kept)))
    return phrase if keep_case else phrase.lower()


def _phrase_from_best_window(
    *,
    tokens: list[str],
    acronym: str,
    bridges: set[str],
    keep_case: bool,
) -> str | None:
    """Build a pruned definition phrase from the best acronym-alignment window.

    Uses the legacy alignment strategy:
      1) Find the smallest token window whose initials can align to `acronym`
         via `_best_window_for_acronym(...)`.
      2) Expand the window to include adjacent numeric-leading tokens (e.g., "3M")
         immediately outside the window.
      3) Prune the window down to:
           - tokens that were part of the alignment (`hit_tokens`),
           - bridge words inside the window (e.g., "of", "and"),
           - numeric-leading tokens (e.g., "3M", "5th") anywhere in the window.
         If pruning removes everything, fall back to the full expanded window.
      4) Collapse whitespace and strip trailing punctuation.

    Args:
        tokens (list[str]): Tokenised label/definition candidate.
        acronym (str): Acronym being aligned (may contain punctuation; alignment helper decides rules).
        bridges (set[str]): Lowercased "bridge" tokens to retain within the window
            even if they are not alignment hits (e.g., {"of", "and"}).
        keep_case (bool): If True, preserves casing from `tokens`. If False, lowercases
            the final phrase.

    Returns:
        str | None: A whitespace-collapsed phrase derived from the best alignment window,
        or None if no alignment window exists.
    """
    win = _best_window_for_acronym(tokens, acronym)
    if not win:
        return None

    i, j, hit_tokens = win

    # Preserve numeric-leading neighbours (e.g. '3M Portable Format' for PF)
    while i > 0 and _numeric_leading(tokens[i - 1]):
        i -= 1
    while j + 1 < len(tokens) and _numeric_leading(tokens[j + 1]):
        j += 1

    kept: list[str] = []
    for idx in range(i, j + 1):
        tok = tokens[idx]
        if idx in hit_tokens or tok.lower() in bridges or _numeric_leading(tok):
            kept.append(tok)

    if not kept:
        kept = tokens[i: j + 1]

    phrase = strip_trailing_punct_str(collapse_ws(" ".join(kept)))
    return phrase if keep_case else phrase.lower()


def tighten_label_by_acronym(
    raw_label: str,
    acronym: str,
    *,
    bridges: Optional[set[str]] = None,
    keep_case: bool = True,
) -> str:
    """
    Canonicalise -> tokenise (compound-aware) -> prefer an initials-in-order window
    for split acronyms (C/A, R&D, A/B/C), keeping bridge words inside the window;
    otherwise fall back to the legacy smallest-window alignment.

    Returns a pruned, whitespace-collapsed phrase (case preserved unless keep_case=False).
    """
    if not raw_label or not acronym:
        out = raw_label or ""
        return out if keep_case else out.lower()

    br = bridges or BRIDGES_DEFAULT

    s = canonicalize(raw_label)
    tokens = _tokenize_preserve(s)
    if not tokens:
        out = strip_trailing_punct_str(collapse_ws(s))
        return out if keep_case else out.lower()

    # 1) Preferred split-acronym path
    preferred = _try_split_acronym_initials_window(tokens=tokens, acronym=acronym, bridges=br, keep_case=keep_case)
    if preferred:
        return preferred

    # 2) Legacy fallback
    legacy = _phrase_from_best_window(tokens=tokens, acronym=acronym, bridges=br, keep_case=keep_case)
    if legacy:
        return legacy

    out = strip_trailing_punct_str(collapse_ws(s))
    return out if keep_case else out.lower()
