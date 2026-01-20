import re
from typing import Mapping

from plainera_unacronym.nlp import FirstOccurrence
from plainera_unacronym.nlp.common.shared import normalize_definition
from plainera_unacronym.nlp.common.types import ExtractedDefinition
from plainera_unacronym.nlp.extraction.config import ExtractionConfig
from plainera_unacronym.nlp.extraction.anchored.normalise import (
    tighten_definition_span,
    collapse_ws,
)
from plainera_unacronym.nlp.extraction.core.collect import initials_match
from plainera_unacronym.nlp.extraction.matchers.helper_patterns import has_letters
from plainera_unacronym.nlp.extraction.matchers.tighten import tighten_label_by_acronym


# Sentence boundary: keep it simple and predictable.
_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?…])\s+|\n+")

_TOKEN_RE = re.compile(r"[A-Za-z0-9][\w’'\-]*")


def _best_span_by_initials(acr: str, sent: str, *, max_chars: int) -> str | None:
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



def _sent_spans(text: str) -> list[tuple[int, int]]:
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


def _find_span_index(spans: list[tuple[int, int]], pos: int) -> int | None:
    for i, (s, e) in enumerate(spans):
        if s <= pos < e:
            return i
    return None


def extract_sentence_backrefs(
    *,
    text: str,
    firsts: Mapping[str, FirstOccurrence],
    cfg: ExtractionConfig,
) -> list[ExtractedDefinition]:
    """
    Extract definitions where the long-form appears in a previous sentence, and the acronym
    appears in the next sentence (e.g., "JSON Web Tokens. JWT is issued ...").

    Returns ExtractedDefinition items (like other extractors) so Flow can merge/dedupe/gapfill.
    """
    max_chars = getattr(cfg, "max_phrase_chars", 200)
    require_two_words = getattr(cfg, "require_two_words", False)

    spans = _sent_spans(text)
    if not spans:
        return []

    out: list[ExtractedDefinition] = []

    for key, fo in firsts.items():
        acr = (key or fo.acronym).upper()

        # Locate which sentence contains the acronym first occurrence
        si = _find_span_index(spans, fo.start_offset)
        if si is None or si == 0:
            continue

        prev_s, prev_e = spans[si - 1]
        prev_raw = text[prev_s:prev_e].strip()
        if not prev_raw:
            continue

        # Gate: don't even try if the previous sentence is absurdly long
        prev_collapsed = collapse_ws(prev_raw)
        if len(prev_collapsed) > max_chars * 3:
            continue

        # Remove trailing punctuation for matching stability
        sent = prev_collapsed.rstrip(" \t\r\n.?!…;:")

        # Find best matching phrase inside the sentence by acronym initials
        cand = _best_span_by_initials(acr, sent, max_chars=max_chars)
        if not cand:
            continue

        # Now run your standard tightening/normalisation (safe on a short span)
        cand = tighten_label_by_acronym(
            cand,
            acr,
            stopwords=set(getattr(cfg, "stop", ())),
            bridges=set(getattr(cfg, "bridges", ())),
        )
        cand = normalize_definition(cand)

        if not cand or not has_letters(cand):
            continue

        # Reject if it collapses to the acronym itself
        if cand.replace(" ", "").upper() == acr.replace(" ", ""):
            continue

        if len(cand) > max_chars:
            continue

        if require_two_words and len(_TOKEN_RE.findall(cand)) < 2:
            continue

        # Validation: must actually match the acronym
        if not initials_match(acr, cand):
            continue

        # We don’t have a tight sub-span of the “cand” inside prev sentence (yet),
        # so start with the whole previous sentence span for def_start/def_end.
        # (You can tighten later by searching for cand within prev_raw.)
        def_start, def_end = prev_s, prev_e

        out.append(
            ExtractedDefinition(
                acronym=acr,
                definition=cand,
                source="in_text",
                confidence=0.50,  # advisory; you can tune later
                acr_start=fo.start_offset,
                acr_end=fo.end_offset,
                def_start=def_start,
                def_end=def_end,
                original_definition=prev_raw,
            )
        )
    return out
