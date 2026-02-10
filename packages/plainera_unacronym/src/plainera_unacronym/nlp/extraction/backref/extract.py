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

from typing import Mapping

from plainera_unacronym.nlp import FirstOccurrence
from plainera_unacronym.nlp.common.constants_regex import TOKEN_RE
from plainera_unacronym.nlp.common.types import ExtractedDefinition, Span
from plainera_unacronym.nlp.extraction.anchored.clean import clean_definition
from plainera_unacronym.nlp.extraction.anchored.normalise import tighten_definition_span
from plainera_unacronym.nlp.extraction.backref.spans import best_span_by_initials, find_span_index, sent_spans
from plainera_unacronym.nlp.extraction.config import ExtractionConfig


def _candidate_from_prev_sentence(
    *,
    acr_norm: str,
    prev_text: str,
    cfg: ExtractionConfig,
    max_chars: int,
    require_two_words: bool,
) -> str | None:
    """Build a validated candidate definition from a previous sentence.

    Prefers a definition-ish span from the previous sentence using `tighten_definition_span`
    and the shared `clean_definition` pipeline. Falls back to an initials-based shortest
    span (`best_span_by_initials`) and cleans it the same way.

    Args:
        acr_norm (str): Normalised acronym (typically uppercased) to match against.
        prev_text (str): Raw previous-sentence slice from the document.
        cfg (ExtractionConfig): Extraction configuration used by `clean_definition`.
        max_chars (int): Maximum allowed candidate length (characters) for span selection.
        require_two_words (bool): If True, require >=2 tokens (enforced post-clean).

    Returns:
        str | None: Cleaned candidate definition if found, otherwise None.
    """
    prev_raw = prev_text.strip()
    if not prev_raw:
        return None

    sent = prev_raw.rstrip(" \t\r\n.?!…;:")

    # 1) Prefer a definition-ish run using the same normaliser as anchored inline.
    base = tighten_definition_span(sent)
    clean = clean_definition(base, acr_norm=acr_norm, cfg=cfg, kind="inline")
    if (
        clean
        and clean.replace(" ", "").upper() != acr_norm.replace(" ", "")
        and (not require_two_words or len(TOKEN_RE.findall(clean)) >= 2)
    ):
        return clean

    # 2) Fallback: initials-based shortest span, then reuse the same cleaner.
    cand = best_span_by_initials(acr_norm, sent, max_chars=max_chars)
    if not cand:
        return None

    clean = clean_definition(cand, acr_norm=acr_norm, cfg=cfg, kind="inline")
    if not clean:
        return None

    if clean.replace(" ", "").upper() == acr_norm.replace(" ", ""):
        return None

    if require_two_words and len(TOKEN_RE.findall(clean)) < 2:
        return None

    return clean


def _find_backref_candidate(
    *,
    text: str,
    spans: list[Span],
    si: int,
    acr_norm: str,
    cfg: ExtractionConfig,
    max_chars: int,
    require_two_words: bool,
) -> tuple[str, Span] | None:
    """Search previous sentence spans for a back-reference definition candidate.

    Looks backwards from the sentence containing an acronym occurrence (sentence index `si`)
    and evaluates up to `cfg.sentence_backref_lookback` previous sentences (nearest first).
    For each prior sentence, delegates to `_candidate_from_prev_sentence` to produce a
    cleaned/validated candidate definition.

    Args:
        text (str): Full document text.
        spans (list[Span]): Sentence-like spans as (start, end) offsets into `text`.
        si (int): Index of the span that contains the acronym occurrence.
        acr_norm (str): Normalised acronym (typically uppercased).
        cfg (ExtractionConfig): Extraction configuration (reads `sentence_backref_lookback`).
        max_chars (int): Maximum allowed candidate length (characters).
        require_two_words (bool): If True, candidate must contain at least two tokens.

    Returns:
        tuple[str, Span] | None: `(candidate, (prev_start, prev_end))` for the first
        previous sentence that yields a candidate, otherwise None.
    """
    sent_lookback = getattr(cfg, "sentence_backref_lookback", 2)

    for back in range(1, min(sent_lookback, si) + 1):
        prev_s, prev_e = spans[si - back]
        prev_slice = text[prev_s:prev_e]

        cand = _candidate_from_prev_sentence(
            acr_norm=acr_norm,
            prev_text=prev_slice,
            cfg=cfg,
            max_chars=max_chars,
            require_two_words=require_two_words,
        )
        if cand:
            return cand, (prev_s, prev_e)

    return None


def _emit_backref_def(
    *,
    acr_norm: str,
    fo: FirstOccurrence,
    cand: str,
    prev_span: Span,
    text: str,
) -> ExtractedDefinition:
    prev_s, prev_e = prev_span
    return ExtractedDefinition(
        acronym=acr_norm,
        definition=cand,
        source="backref",
        definition_confidence=0.50,
        acr_start=fo.start_offset,
        acr_end=fo.end_offset,
        def_start=prev_s,
        def_end=prev_e,
        original_definition=text[prev_s:prev_e].strip(),
        kind="sentence_backref",
    )


def _alpha_len(s: str) -> int:
    """Count alphabetic characters in a string.

    Non-letter characters (digits, punctuation, whitespace) are ignored. Uses
    `str.isalpha()` so Unicode letters are counted as well.

    Args:
        s (str): Input string.

    Returns:
        int: Number of alphabetic characters in `s`.
    """
    return sum(1 for c in s if c.isalpha())


def extract_sentence_backrefs(
    *,
    text: str,
    firsts: Mapping[str, FirstOccurrence],
    cfg: ExtractionConfig,
) -> list[ExtractedDefinition]:
    """Extract Tier-1 sentence back-reference definitions.

    For each first occurrence, locate the sentence containing the acronym and
    search previous sentence(s) for a plausible long-form candidate. Emits a
    backref definition when a candidate passes backref guardrails.

    Args:
        text (str): Full document text.
        firsts (Mapping[str, FirstOccurrence]): Normalised key -> first occurrence.
        cfg (ExtractionConfig): Extraction configuration.

    Returns:
        list[ExtractedDefinition]: Zero or more backref definitions.
    """
    max_chars = getattr(cfg, "max_phrase_chars", 200)
    require_two_words = getattr(cfg, "sentence_backref_require_two_words", True)

    spans = sent_spans(text)
    if not spans:
        return []

    out: list[ExtractedDefinition] = []

    for key, fo in firsts.items():
        acr_norm = (fo.normalized_key or key or fo.acronym).upper()
        acr_alpha_len = sum(1 for c in acr_norm if c.isalpha())
        if acr_alpha_len < cfg.min_acr_len:
            continue

        si = find_span_index(spans, fo.start_offset)
        if si is None or si == 0:
            continue

        hit = _find_backref_candidate(
            text=text,
            spans=spans,
            si=si,
            acr_norm=acr_norm,
            cfg=cfg,
            max_chars=max_chars,
            require_two_words=require_two_words,
        )
        if not hit:
            continue

        cand, prev_span = hit
        out.append(
            _emit_backref_def(
                acr_norm=acr_norm,
                fo=fo,
                cand=cand,
                prev_span=prev_span,
                text=text,
            )
        )

    return out
