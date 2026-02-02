import re

from plainera_unacronym.nlp.common.shared import collapse_ws
from plainera_unacronym.nlp.common.types import Span


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

    Splits `sent` on whitespace into tokens and builds token initials using the first
    character of each token when alphabetic. Then searches for the shortest contiguous
    token window whose initials match the alphabetic characters of `acr` in order.
    The returned span is whitespace-collapsed via `collapse_ws` and must be <= `max_chars`.

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

    for i in range(len(tokens)):
        ai = 0  # index into A
        for j in range(i, len(tokens)):
            init = tok_inits[j]
            if not init:
                continue

            # consume as many letters of `init` as match the acronym tail
            k = 0
            while ai < len(A) and k < len(init) and init[k] == A[ai]:
                ai += 1
                k += 1

            if ai == len(A):
                cand = collapse_ws(" ".join(tokens[i: j + 1]).strip())
                if cand and len(cand) <= max_chars:
                    if best is None:
                        best = (i, j)
                    else:
                        bi, bj = best
                        if (j - i) < (bj - bi) or (
                            (j - i) == (bj - bi) and len(cand) < len(" ".join(tokens[bi: bj + 1]))
                        ):
                            best = (i, j)
                break  # for this i, smallest j wins

    if best is None:
        return None

    i, j = best
    out = collapse_ws(" ".join(tokens[i: j + 1]).strip())
    return out if out else None


def sent_spans(text: str) -> list[Span]:
    """Split text into sentence-ish spans using a simple boundary regex.

    Sentence boundaries are detected by `_SENT_BOUNDARY_RE` (punctuation followed by
    whitespace, or one-or-more newlines). Returned spans are `(start, end)` offsets
    into the original `text` (end-exclusive). Empty spans are not returned.

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

    A position is considered inside a span when `start <= pos < end`
    (end-exclusive).

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
