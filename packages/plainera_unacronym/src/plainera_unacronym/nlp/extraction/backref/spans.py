import re

from plainera_unacronym.nlp.common.shared import collapse_ws

# Sentence boundary: keep it simple and predictable.
_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?…])\s+|\n+")

def best_span_by_initials(acr: str, sent: str, *, max_chars: int) -> str | None:
    """
    Find the shortest contiguous token span in `sent` whose initials match `acr`.
    Returns the span text (whitespace-collapsed), or None.
    """
    tokens = [t for t in sent.split() if t]
    if not tokens:
        return None

    # Precompute initials for each token (ignore tokens starting with non-alpha)
    tok_inits = [t[0].upper() if t and t[0].isalpha() else "" for t in tokens]
    A = [c.upper() for c in acr if c.isalpha()]
    if not A:
        return None

    best: tuple[int, int] | None = None  # (i,j) inclusive span

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

def sent_spans(text: str) -> list[tuple[int, int]]:
    """Return (start,end) spans for sentence-ish chunks."""
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _SENT_BOUNDARY_RE.finditer(text):
        end = m.start()
        if end > start:
            spans.append((start, end))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def find_span_index(spans: list[tuple[int, int]], pos: int) -> int | None:
    for i, (s, e) in enumerate(spans):
        if s <= pos < e:
            return i
    return None
