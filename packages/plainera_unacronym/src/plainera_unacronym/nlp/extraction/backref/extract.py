"""
    Tier-1 “sentence back-reference” extractor.

    Purpose
    -------
    This stage exists to catch the pattern where a definition appears in a *previous* sentence and the acronym
    appears later without an inline/parenthetical definition.

        Example:
            "We use Single sign-on for authentication. SSO is enabled by default."

    It is explicitly *not* a parenthetical extractor. It does not attempt to parse:
        - "Long Form (ACR)"
        - "ACR (Long Form)"
        - "Long Form - ACR"
    Those are handled by the anchored/harvest stages.

    How it works (high level)
    -------------------------
    For each acronym first-occurrence (from the detector):
      1) Split the document into “sentence-ish” spans using a simple regex boundary.
      2) Locate the sentence span that contains the acronym occurrence.
      3) Look backwards across up to N previous sentences (cfg.sentence_backref_lookback, default=2).
      4) For each candidate previous sentence:
            a) Find the shortest contiguous token span whose initials match the acronym
               (via _best_span_by_initials).
            b) Tighten/normalise the candidate label (tighten_label_by_acronym + normalize_definition).
            c) Validate using guardrails:
                  - must contain letters
                  - must not be identical to the acronym itself
                  - must be <= cfg.max_phrase_chars
                  - optionally require >=2 tokens (cfg.require_two_words)
                  - must pass initials_match(acronym, candidate)
            d) The first valid match wins (nearest previous sentence first).
      5) Emit an ExtractedDefinition with source="backref" and kind="sentence_backref".

    Important behavioural constraints
    --------------------------------
    - Only looks *backwards* across sentence boundaries.
      If the acronym appears in the first sentence (sentence index 0), this stage will never fire.
      That is by design: it prevents large, noisy “document-wide” hunting.

    - Sentence segmentation is intentionally conservative and predictable.
      It uses punctuation/newlines as boundaries, not a full NLP sentence model.

    - This stage is intended to be deterministic and high-precision.
      If it cannot find a mechanically defensible initials span in the immediate prior sentence(s),
      it returns no result rather than guessing.

    Config knobs
    ------------
    - cfg.sentence_backref_lookback:
        How many previous sentences to search (nearest-first). Default is 2.

    - cfg.max_phrase_chars:
        Maximum character length allowed for a candidate definition span.

    - cfg.require_two_words:
        If true, candidate must contain at least two tokens as defined by _TOKEN_RE.

    Notes
    -----
    - Acronym matching should preserve the detector’s acronym casing for output, but may use
      uppercasing internally for comparison. If you change casing behaviour, keep Tier-1
      invariants: “do not rewrite the user’s acronym token”.

    Returns
    -------
    list[ExtractedDefinition]
        Zero or more extracted definitions; each corresponds to an acronym whose definition
        was found in a prior sentence using initials-based span selection.
    """


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
