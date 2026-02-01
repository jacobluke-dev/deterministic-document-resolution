import re

from plainera_unacronym.nlp.common.shared import collapse_ws
from plainera_unacronym.nlp.common.types import Span

# Sentence boundary: keep it simple and predictable.
_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?…])\s+|\n+")

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

    # Precompute initials for each token (ignore tokens starting with non-alpha)
    tok_inits = [t[0].upper() if t and t[0].isalpha() else "" for t in tokens]
    A = [c.upper() for c in acr if c.isalpha()]
    if not A:
        return None

    best: Span | None = None  # (i,j) inclusive span

    for i in range(len(tokens)):
        ai = 0
        for j in range(i, len(tokens)):
            if tok_inits[j] and tok_inits[j] == A[ai]:
                ai += 1
                if ai == len(A):
                    # candidate span found: minimise length (j-i), then chars
                    cand = " ".join(tokens[i : j + 1]).strip()
                    cand = collapse_ws(cand)
                    if len(cand) <= max_chars:
                        if best is None:
                            best = (i, j)
                        else:
                            bi, bj = best
                            # prefer fewer tokens, then fewer chars
                            if (j - i) < (bj - bi):
                                best = (i, j)
                            elif (j - i) == (bj - bi) and len(cand) < len(" ".join(tokens[bi : bj + 1])):
                                best = (i, j)
                    break  # for this i, smallest j already

    if best is None:
        return None

    i, j = best
    out = " ".join(tokens[i : j + 1]).strip()
    out = collapse_ws(out)
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
