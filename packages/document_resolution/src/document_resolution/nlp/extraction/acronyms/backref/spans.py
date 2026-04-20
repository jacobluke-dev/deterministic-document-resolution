import re

from document_resolution.nlp.common.shared import collapse_ws
from document_resolution.nlp.common.types import Span

# Sentence boundary: keep it simple and predictable.
_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?…])\s+|\n+")
# Split on ASCII hyphen, unicode hyphen/dashes, and slash.
_SUBTOK_SPLIT_RE = re.compile(r"[-\u2010\u2011\u2012\u2013\u2014/]+")
_EDGE_PUNCT_RE = re.compile(r"^[^\w]+|[^\w]+$")


def token_initials(token: str) -> str:
    """Return initials contributed by a token, including hyphenated subparts.

    Examples:
        token_initials("Single") -> "S"
        token_initials("sign-on") -> "SO"
        token_initials("3M") -> ""   (ignored; non-alpha leading)
        token_initials("(SSO)") -> "S" (after trimming edges; still only first subpart)
    """
    if not token:
        return ""

    t = _EDGE_PUNCT_RE.sub("", token)
    if not t or not t[0].isalpha():
        return ""

    parts = [p for p in _SUBTOK_SPLIT_RE.split(t) if p]
    out: list[str] = []
    for p in parts:
        p = _EDGE_PUNCT_RE.sub("", p)
        if p and p[0].isalpha():
            out.append(p[0].upper())
    return "".join(out)


def best_span_by_initials(acr: str, sent: str, *, max_chars: int) -> str | None:
    """Find the shortest contiguous token span whose initials match an acronym.

    Args:
        acr (str): Acronym to match. Non-letters are ignored.
        sent (str): Sentence text to search within.
        max_chars (int): Maximum allowed length of the returned span (in characters).

    Returns:
        str | None: The best-matching span text (whitespace-collapsed) if found,
        otherwise None.
    """
    tokens = [t for t in sent.split() if t]
    if not tokens:
        return None

    A = [c.upper() for c in acr if c.isalpha()]
    if not A:
        return None

    tok_inits = [token_initials(t) for t in tokens]  # may be multi-letter, e.g. "SO"

    best: Span | None = None  # (i, j) inclusive
    best_len = 10**9
    best_chars = 10**9

    for i in range(len(tokens)):
        j = _best_window_end_for_initials(A, tok_inits, i)
        if j is None:
            continue

        cand, cand_chars = _candidate_span(tokens, i, j)
        if not cand or cand_chars > max_chars:
            continue

        win_len = j - i
        if win_len < best_len or (win_len == best_len and cand_chars < best_chars):
            best = (i, j)
            best_len = win_len
            best_chars = cand_chars

    if best is None:
        return None

    i, j = best
    out = collapse_ws(" ".join(tokens[i : j + 1]).strip())
    return out if out else None


def _best_window_end_for_initials(
    A: list[str],
    tok_inits: list[str],
    start_idx: int,
) -> int | None:
    """Return the smallest end index j >= i such that initials from i..j match A.

    Returns:
        int | None: The minimal end index `j` achieving a full match, else None.
    """
    ai = 0
    for j in range(start_idx, len(tok_inits)):
        init = tok_inits[j]
        if not init:
            continue

        k = 0
        while ai < len(A) and k < len(init) and init[k] == A[ai]:
            ai += 1
            k += 1

        if ai == len(A):
            return j

    return None


def _candidate_span(tokens: list[str], start_idx: int, end_idx: int) -> tuple[str, int]:
    """Build a collapsed candidate span string and return it with its character length.

    Returns:
        tuple[str, int]: (collapsed_span, len(collapsed_span)).
    """
    cand = collapse_ws(" ".join(tokens[start_idx: end_idx + 1]).strip())
    return cand, len(cand) if cand else 0


def sent_spans(text: str) -> list[Span]:
    """Split text into sentence-ish spans using a simple boundary regex.

    Args:
        text (str): Source text to split.

    Returns:
        list[Span]: List of `(start, end)` spans (end-exclusive) covering the text.
    """
    spans: list[Span] = []
    start = 0
    for m in _SENT_BOUNDARY_RE.finditer(text):
        end = m.start()
        if end > start:
            spans.append((start, end))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def find_span_index(spans: list[Span], pos: int) -> int | None:
    """Find the index of the span that contains a position.

    Args:
        spans (list[Span]): List of `(start, end)` spans (end-exclusive).
        pos (int): Position to locate within the spans.

    Returns:
        int | None: Index of the first span containing `pos`, or None if `pos`
        is not contained in any span.
    """
    for i, (s, e) in enumerate(spans):
        if s <= pos < e:
            return i
    return None
