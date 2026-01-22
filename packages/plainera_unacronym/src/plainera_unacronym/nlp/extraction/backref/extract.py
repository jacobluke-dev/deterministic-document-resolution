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


def extract_sentence_backrefs(*, text: str, firsts: Mapping[str, FirstOccurrence], cfg: ExtractionConfig) -> list[ExtractedDefinition]:
    max_chars = getattr(cfg, "max_phrase_chars", 200)
    require_two_words = getattr(cfg, "require_two_words", False)

    sent_lookback = getattr(cfg, "sentence_backref_lookback", 2)

    spans = _sent_spans(text)
    if not spans:
        return []

    out: list[ExtractedDefinition] = []

    for key, fo in firsts.items():
        acr = (key or fo.acronym).upper()

        si = _find_span_index(spans, fo.start_offset)
        if si is None or si == 0:
            continue

        best_cand: str | None = None
        best_prev_span: tuple[int, int] | None = None

        # NEW: try sentence si-1, si-2, ... up to sent_lookback
        for back in range(1, min(sent_lookback, si) + 1):
            prev_s, prev_e = spans[si - back]
            prev_raw = text[prev_s:prev_e].strip()
            if not prev_raw:
                continue

            prev_collapsed = collapse_ws(prev_raw)
            if len(prev_collapsed) > max_chars * 3:
                continue

            sent = prev_collapsed.rstrip(" \t\r\n.?!…;:")

            cand = _best_span_by_initials(acr, sent, max_chars=max_chars)
            if not cand:
                continue

            cand = tighten_label_by_acronym(
                cand,
                acr,
                stopwords=set(getattr(cfg, "stop", ())),
                bridges=set(getattr(cfg, "bridges", ())),
            )
            cand = normalize_definition(cand)

            if not cand or not has_letters(cand):
                continue
            if cand.replace(" ", "").upper() == acr.replace(" ", ""):
                continue
            if len(cand) > max_chars:
                continue
            if require_two_words and len(_TOKEN_RE.findall(cand)) < 2:
                continue
            if not initials_match(acr, cand):
                continue

            # first valid match wins because we’re scanning nearest-first
            best_cand = cand
            best_prev_span = (prev_s, prev_e)
            break

        if not best_cand or not best_prev_span:
            continue

        prev_s, prev_e = best_prev_span

        out.append(
            ExtractedDefinition(
                acronym=acr,
                definition=best_cand,
                source="backref",
                confidence=0.50,
                acr_start=fo.start_offset,
                acr_end=fo.end_offset,
                def_start=prev_s,
                def_end=prev_e,
                original_definition=text[prev_s:prev_e].strip(),
                kind="sentence_backref",
            )
        )

    return out
