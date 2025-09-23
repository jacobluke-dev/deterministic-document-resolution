import re
from typing import Optional

from plainera_unacronym.nlp.common.constants import DEFAULT_STOPWORDS, BRIDGES_DEFAULT
from plainera_unacronym.nlp.common.shared import canonicalize, collapse_ws, strip_trailing_punct


_word_re = re.compile(r"[A-Za-z0-9'’\-\/&\.]+", flags=re.UNICODE)


def _split_compound(token: str) -> list[str]:
    """Split hyphen/slash/dot compounds for initial extraction."""
    # Keep only alpha/num for initials; treat .,/,-,& as boundaries
    parts = re.split(r"[\-\/\.\&]", token)
    return [p for p in parts if p]


def _tokenize_preserve(text: str) -> list[str]:
    return _word_re.findall(text)


def _initials_seq(tokens: list[str], stopwords: set[str]) -> tuple[list[str], list[int]]:
    """
    Build a sequence of initials (letters+digits) from tokens, skipping stopwords.
    owners[k] = token index that produced letters[k].
    """
    letters, owners = [], []
    for ti, tok in enumerate(tokens):
        if tok.lower() in stopwords:
            continue
        for part in _split_compound(tok):
            # initial = first alnum in the part
            m = re.search(r"[A-Za-z0-9]", part)
            if m:
                letters.append(m.group(0).upper())
                owners.append(ti)
    return letters, owners

def _match_from(letters: list[str], A: list[str], start: int) -> Optional[tuple[int, list[int]]]:
    """
    Greedily align A as a subsequence of letters starting at index `start`.
    Returns (end_index_exclusive_in_letters, matched_letter_positions) or None.
    """
    li, ai = start, 0
    used = []
    while li < len(letters) and ai < len(A):
        if letters[li] == A[ai]:
            used.append(li)
            ai += 1
        li += 1
    if ai == len(A):
        return li, used
    return None


# def _best_window_for_acronym(tokens: list[str], acronym: str, stopwords: set) -> Optional[tuple[int, int]]:
#     """
#     Find the shortest contiguous token window whose (non-stopword) initials
#     match the acronym as a subsequence in order. Returns (start_idx, end_idx_inclusive) or None.
#     """
#     A = [c for c in acronym if c.isalnum()]
#     if not A:
#         return None
#     A = [c.upper() for c in A]
#
#     letters, owners = _initials_seq(tokens, stopwords)
#     if not letters:
#         return None
#
#     best = None  # (tok_start, tok_end)
#     nL, nA = len(letters), len(A)
#
#     # Slide over the initials sequence; for each possible start, greedily match A
#     for li in range(nL):
#         ai = 0
#         lj = li
#         while lj < nL and ai < nA:
#             if letters[lj] == A[ai]:
#                 ai += 1
#             lj += 1
#         if ai == nA:
#             tok_start = owners[li]
#             tok_end = owners[lj - 1]
#             if best is None or (tok_end - tok_start) < (best[1] - best[0]):
#                 best = (tok_start, tok_end)
#
#     return best


def _best_window_for_acronym(tokens: list[str], acronym: str, stopwords: set[str]
    ) -> Optional[tuple[int, int, set[int]]]:
    """
    Return (tok_start, tok_end_inclusive, hit_token_indices_set) for the *shortest contiguous*
    token window whose (non-stopword, compound-aware) initials match `acronym` in order.
    """
    A = [c.upper() for c in acronym if c.isalnum()]
    if not A:
        return None

    letters, owners = _initials_seq(tokens, stopwords)
    if not letters:
        return None

    best = None  # (tok_s, tok_e, hits_set)
    for li in range(len(letters)):
        res = _match_from(letters, A, li)
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


def tighten_label_by_acronym(
    raw_label: str,
    acronym: str,
    *,
    stopwords: Optional[set[str]] = None,
    bridges: Optional[set[str]] = None,
    keep_case: bool = True,
) -> str:
    """
    Canonicalise -> tokenise (compound-aware) -> find the shortest contiguous window that
    aligns to the acronym (ignoring stopwords) -> *inside that window* keep only tokens
    that contributed to the match plus any `bridges`. Fallback to canon+collapse+strip.
    """
    if not raw_label or not acronym:
        return raw_label or ""

    stop = stopwords or DEFAULT_STOPWORDS
    br = bridges or BRIDGES_DEFAULT

    s = canonicalize(raw_label)  # preserves case, normalises look-alikes
    tokens = _tokenize_preserve(s)
    if not tokens:
        return strip_trailing_punct(collapse_ws(s))

    win = _best_window_for_acronym(tokens, acronym, stop)
    if not win:
        out = strip_trailing_punct(collapse_ws(s))
        return out if keep_case else out.lower()

    i, j, hit_tokens = win

    # prune within the chosen span: keep matched tokens + bridge words
    kept: list[str] = []
    for idx in range(i, j + 1):
        tok = tokens[idx]
        if idx in hit_tokens or tok.lower() in br:
            kept.append(tok)

    # if pruning removed everything (edge case), keep the original span
    if not kept:
        kept = tokens[i:j + 1]

    phrase = " ".join(kept)
    phrase = strip_trailing_punct(collapse_ws(phrase))
    return phrase if keep_case else phrase.lower()
