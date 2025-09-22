import re
from typing import List, Tuple, Optional

from plainera_unacronym.nlp.common.shared import canonicalize, collapse_ws, strip_trailing_punct

DEFAULT_STOPWORDS = {
    # english-ish function words—expand as needed
    "of", "and", "the", "for", "to", "in", "on", "with", "a", "an", "at", "by", "from", "as", "per",
    # a few common non-English determiners/preps for names
    "de", "la", "le", "du", "des", "del", "da", "di", "von", "und"
}

_word_re = re.compile(r"[A-Za-z0-9'’\-\/&\.]+", flags=re.UNICODE)


def _split_compound(token: str) -> List[str]:
    """Split hyphen/slash/dot compounds for initial extraction."""
    # Keep only alpha/num for initials; treat .,/,-,& as boundaries
    parts = re.split(r"[\-\/\.\&]", token)
    return [p for p in parts if p]


def _tokenize_preserve(text: str) -> List[str]:
    return _word_re.findall(text)


def _initials_seq(tokens: List[str], stopwords: set) -> Tuple[List[str], List[int]]:
    """
    Build a flat sequence of initials (uppercased) and a parallel map of which token each initial came from.
    Stopwords contribute no initial. Compounds contribute multiple initials (e.g., 'Field-Programmable' -> F,P).
    """
    letters, owners = [], []
    for ti, tok in enumerate(tokens):
        low = tok.lower()
        if low in stopwords:
            continue
        for part in _split_compound(tok):
            # the initial is the first alphabetical character in the part
            for ch in part:
                if ch.isalpha():
                    letters.append(ch.upper())
                    owners.append(ti)
                    break
    return letters, owners


def _best_window_for_acronym(tokens: List[str], acronym: str, stopwords: set) -> Optional[Tuple[int, int]]:
    """
    Find the shortest contiguous token window whose (non-stopword) initials
    match the acronym as a subsequence in order. Returns (start_idx, end_idx_inclusive) or None.
    """
    A = [c for c in acronym if c.isalnum()]
    if not A:
        return None
    A = [c.upper() for c in A]

    letters, owners = _initials_seq(tokens, stopwords)
    if not letters:
        return None

    best = None  # (tok_start, tok_end)
    nL, nA = len(letters), len(A)

    # Slide over the initials sequence; for each possible start, greedily match A
    for li in range(nL):
        ai = 0
        lj = li
        while lj < nL and ai < nA:
            if letters[lj] == A[ai]:
                ai += 1
            lj += 1
        if ai == nA:
            tok_start = owners[li]
            tok_end = owners[lj - 1]
            if best is None or (tok_end - tok_start) < (best[1] - best[0]):
                best = (tok_start, tok_end)

    return best


def tighten_label_by_acronym(raw_label: str, acronym: str, stopwords=None) -> str:
    """
    Return the minimal contiguous phrase inside `raw_label` whose initials (ignoring stopwords)
    match `acronym` in order. If no alignment is found, fall back to your current
    normalization pipeline.
    """
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS
    if not raw_label or not acronym:
        return raw_label or ""

    # Canonicalize like the rest of your pipeline
    s = canonicalize(raw_label)
    tokens = _tokenize_preserve(s)

    win = _best_window_for_acronym(tokens, acronym, stopwords)
    if win is not None:
        i, j = win
        # Reconstruct exact surface from tokens i..j using original spacing:
        #  - join tokens with single space, then run your normal strip/collapse.
        phrase = " ".join(tokens[i:j + 1])
        return strip_trailing_punct(collapse_ws(phrase))

    # Fallback: your prior tightening (canonicalize+collapse+strip)
    return strip_trailing_punct(collapse_ws(s))
