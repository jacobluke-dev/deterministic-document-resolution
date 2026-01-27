import re
from typing import Optional

from plainera_unacronym.nlp.common.constants_regex import BRIDGES_DEFAULT
from plainera_unacronym.nlp.common.shared import canonicalize, strip_trailing_punct_str, collapse_ws
from plainera_unacronym.nlp.extraction.matchers.common import match_from, initials_seq, is_mixed_case_acronym

_word_re = re.compile(r"[A-Za-z0-9'’\-\/&\.]+", flags=re.UNICODE)


def _tokenize_preserve(text: str) -> list[str]:
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

    def _numeric_leading(tok: str) -> bool:
        for ch in tok:
            if ch.isalnum():
                return not ch.isalpha()
        return False

    if not raw_label or not acronym:
        return raw_label or ""

    br = bridges or BRIDGES_DEFAULT

    s = canonicalize(raw_label)  # preserves case, normalises look-alikes
    tokens = _tokenize_preserve(s)
    if not tokens:
        return strip_trailing_punct_str(collapse_ws(s))
    # --- NEW: prefer initials-in-order for split acronyms (e.g., "C/A") ---
    # Extract alnum letters from acronym, in order.
    letters = [c for c in re.sub(r"[^A-Za-z0-9]+", "", acronym).upper()]
    if len(letters) >= 2:
        # Find the smallest window whose token-initials contain the letters in order.
        seq_idxs: list[int] = []
        li = 0
        for idx, tok in enumerate(tokens):
            ch = tok[0].upper() if tok else ""
            if ch == letters[li]:
                seq_idxs.append(idx)
                li += 1
                if li == len(letters):
                    break

        if len(seq_idxs) == len(letters):
            low, high = min(seq_idxs), max(seq_idxs)

            # deals with numerical leading acronyms/tokens e.g. 3M
            while low > 0 and _numeric_leading(tokens[low - 1]):
                low -= 1
            while high + 1 < len(tokens) and _numeric_leading(tokens[high + 1]):
                high += 1

            # Build kept: keep matched-initial tokens, plus any bridge words within the window.
            kept: list[str] = []
            hit_set = set(seq_idxs)
            for idx in range(low, high + 1):
                tok = tokens[idx]
                if idx in hit_set or tok.lower() in br or _numeric_leading(tok):
                    kept.append(tok)

            # If pruning removed everything, keep the full window.
            if not kept:
                kept = tokens[low: high + 1]

            phrase = " ".join(kept)
            phrase = strip_trailing_punct_str(collapse_ws(phrase))

            _SPLIT_ACR_RE = re.compile(r"[&./-]")  # treat these as “split” markers
            is_split = bool(_SPLIT_ACR_RE.search(acronym))
            if is_split:
                letters = [c for c in re.sub(r"[^A-Za-z0-9]+", "", acronym).upper()]
                if len(letters) >= 2:
                    ...  # your existing “initials-in-order” block
                    return phrase if keep_case else phrase.lower()
            return phrase if keep_case else phrase.lower()

    # Legacy path: choose smallest window aligned to the acronym (ignoring stopwords).
    win = _best_window_for_acronym(tokens, acronym)
    if not win:
        out = strip_trailing_punct_str(collapse_ws(s))
        return out if keep_case else out.lower()

    i, j, hit_tokens = win

    # prune within the chosen span: keep matched tokens + bridge words
    kept: list[str] = []
    for idx in range(i, j + 1):
        tok = tokens[idx]
        if idx in hit_tokens or tok.lower() in br:
            kept.append(tok)

    if not kept:
        kept = tokens[i: j + 1]

    phrase = " ".join(kept)
    phrase = strip_trailing_punct_str(collapse_ws(phrase))
    return phrase if keep_case else phrase.lower()
