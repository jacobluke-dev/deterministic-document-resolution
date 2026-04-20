import re
from typing import Optional

from document_resolution.nlp.common.constants_regex import BRIDGES_DEFAULT
from document_resolution.nlp.common.shared import canonicalize, collapse_ws, strip_trailing_punct_str
from document_resolution.nlp.extraction.acronyms.matchers.common import initials_seq, is_mixed_case_acronym, match_from

_word_re = re.compile(r"[A-Za-z0-9'’\-\/&\.]+", flags=re.UNICODE)


def _tokenize_preserve(text: str) -> list[str]:
    """Tokenise `text` into “word-ish” tokens while preserving certain punctuation.

    Args:
        text (str): Input string to tokenise.

    Returns:
        list[str]: List of matched tokens (possibly empty).
    """
    return _word_re.findall(text)


def _best_window_for_acronym(tokens: list[str], acronym: str) -> Optional[tuple[int, int, set[int]]]:
    """Select the shortest contiguous token window whose initials match an acronym.

    Args:
        tokens (list[str]): Token strings to search within.
        acronym (str): Acronym whose alphanumeric characters must match in order.

    Returns:
        Optional[tuple[int, int, set[int]]]: A tuple `(tok_start, tok_end_inclusive, hits)`
        where `tok_start`/`tok_end_inclusive` bound the chosen window in `tokens` and
        `hits` is the set of token indices that contributed matched initials. Returns
        None if no match is possible (e.g., empty acronym or no initials produced).
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

    Scans LTR for the first alphanumeric character.

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

    Args:
        tokens (list[str]): Tokenised phrase (case-preserving).
        acronym (str): Acronym surface form, used to derive the initials sequence.
        bridges (set[str]): Lowercased bridge words to retain when inside the matched window.
        keep_case (bool): If True, preserve original casing in the returned phrase; otherwise
            return a lowercased phrase.

    Returns:
        str | None: The tightened phrase if a match is found, otherwise ``None``.
    """
    letters = _split_acr_letters(acronym)
    if letters is None:
        return None

    seq_idxs = _match_initials_subsequence(tokens, letters)
    if seq_idxs is None:
        return None

    low, high = _expand_numeric_neighbours(tokens, min(seq_idxs), max(seq_idxs))

    kept = _collect_kept_tokens(tokens, seq_idxs, low, high, bridges)
    phrase = strip_trailing_punct_str(collapse_ws(" ".join(kept)))
    return phrase if keep_case else phrase.lower()


def _split_acr_letters(acronym: str) -> list[str] | None:
    """Return cleaned acronym letters for split-marker acronyms.

    Args:
        acronym (str): Acronym surface form.

    Returns:
        list[str] | None: Uppercased alphanumeric characters, or None if not applicable.
    """
    if not _SPLIT_ACR_MARKER_RE.search(acronym):
        return None

    letters = [c for c in re.sub(r"[^A-Za-z0-9]+", "", acronym).upper()]
    return letters if len(letters) >= 2 else None


def _match_initials_subsequence(tokens: list[str], letters: list[str]) -> list[int] | None:
    """Find the first token-index subsequence whose initials match `letters` in order.

    Args:
        tokens (list[str]): Tokenised phrase.
        letters (list[str]): Target initials sequence to match.

    Returns:
        list[int] | None: Token indices participating in the match, or None if no full match.
    """
    seq_idxs: list[int] = []
    li = 0
    for idx, tok in enumerate(tokens):
        ch = tok[0].upper() if tok else ""
        if ch == letters[li]:
            seq_idxs.append(idx)
            li += 1
            if li == len(letters):
                return seq_idxs
    return None


def _expand_numeric_neighbours(tokens: list[str], low: int, high: int) -> tuple[int, int]:
    """Expand [low, high] to include numeric-leading neighbours immediately adjacent.

    Args:
        tokens (list[str]): Tokenised phrase.
        low (int): Initial low bound (inclusive).
        high (int): Initial high bound (inclusive).

    Returns:
        tuple[int, int]: Expanded (low, high) bounds (inclusive).
    """
    while low > 0 and _numeric_leading(tokens[low - 1]):
        low -= 1
    while high + 1 < len(tokens) and _numeric_leading(tokens[high + 1]):
        high += 1
    return low, high


def _collect_kept_tokens(
    tokens: list[str],
    seq_idxs: list[int],
    low: int,
    high: int,
    bridges: set[str],
) -> list[str]:
    """Collect tokens to keep from a matched window.

    Falls back to returning the full window if the keep-set is empty.

    Args:
        tokens (list[str]): Tokenised phrase.
        seq_idxs (list[int]): Matched token indices.
        low (int): Window low bound (inclusive).
        high (int): Window high bound (inclusive).
        bridges (set[str]): Lowercased bridge tokens to retain.

    Returns:
        list[str]: Tokens to join into the phrase.
    """
    hit_set = set(seq_idxs)
    kept: list[str] = []
    for idx in range(low, high + 1):
        tok = tokens[idx]
        if idx in hit_set or tok.lower() in bridges or _numeric_leading(tok):
            kept.append(tok)

    return kept if kept else tokens[low : high + 1]


def _phrase_from_best_window(
    *,
    tokens: list[str],
    acronym: str,
    bridges: set[str],
    keep_case: bool,
) -> str | None:
    """Build a pruned definition phrase from the best acronym-alignment window.

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
        kept = tokens[i : j + 1]

    phrase = strip_trailing_punct_str(collapse_ws(" ".join(kept)))
    return phrase if keep_case else phrase.lower()


def tighten_label_by_acronym(
    raw_label: str,
    acronym: str,
    *,
    bridges: Optional[set[str]] = None,
    keep_case: bool = True,
) -> str:
    """Tighten a candidate definition by aligning it to an acronym.

    This function does *not* implement stopword logic beyond the optional `bridges`
    set. Any stopword filtering used to compute initials/alignments is handled by
    lower-level helpers (e.g. token/initial generation and matching).

    Args:
        raw_label (str): Candidate long-form label/definition extracted from text.
        acronym (str): Acronym used to align and prune the label.
        bridges (Optional[set[str]]): Lowercased "bridge" words that may be retained
            inside a chosen window even if they are not hit tokens (e.g. {"of", "and"}).
            If None, defaults to `BRIDGES_DEFAULT`.
        keep_case (bool): If True, preserve the original casing of the retained tokens.
            If False, return the tightened phrase lowercased.

    Returns:
        str: A whitespace-collapsed, punctuation-trimmed phrase tightened around the
        acronym alignment. If no alignment is possible, returns the canonicalised
        label (trimmed/collapsed) as a conservative fallback.
    """
    if not raw_label or not acronym:
        out = raw_label or ""
        return out if keep_case else out.lower()

    br: set[str] = set(bridges) if bridges is not None else set(BRIDGES_DEFAULT)

    s = canonicalize(raw_label)
    tokens = _tokenize_preserve(s)
    if not tokens:
        out = strip_trailing_punct_str(collapse_ws(s))
        return out if keep_case else out.lower()

    # 1) Preferred split-acronym path
    preferred = _try_split_acronym_initials_window(tokens=tokens, acronym=acronym, bridges=br, keep_case=keep_case)
    if preferred:
        return preferred

    # 2) fallback
    fallback = _phrase_from_best_window(tokens=tokens, acronym=acronym, bridges=br, keep_case=keep_case)
    if fallback:
        return fallback

    out = strip_trailing_punct_str(collapse_ws(s))
    return out if keep_case else out.lower()
