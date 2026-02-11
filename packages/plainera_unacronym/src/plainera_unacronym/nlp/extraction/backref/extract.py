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

from typing import Mapping, Literal

from plainera_unacronym.nlp import FirstOccurrence
from plainera_unacronym.nlp.common.constants_regex import TOKEN_RE, HYPHEN_SPLIT_RE
from plainera_unacronym.nlp.common.types import ExtractedDefinition, Span
from plainera_unacronym.nlp.extraction.anchored.clean import clean_definition
from plainera_unacronym.nlp.extraction.anchored.normalise import tighten_definition_span
from plainera_unacronym.nlp.extraction.backref.spans import best_span_by_initials, find_span_index, sent_spans
from plainera_unacronym.nlp.extraction.config import ExtractionConfig
from plainera_unacronym.nlp.extraction.core.collect import initials_match
from plainera_unacronym.nlp.extraction.engine.confidence import base_conf_for

BackrefEvidence = Literal["definitionish", "initials"]


def _initials_hyphen_aware(phrase: str) -> str:
    """
    Build an initials string from a phrase, treating hyphenated tokens as multiple parts.
    Useful for matching acronyms against phrases like "Single sign-on" → "SSO".

    Args:
        phrase: Input phrase.

    Returns:
        Uppercased initials string.
    """
    letters: list[str] = []
    for tok in TOKEN_RE.findall(phrase):
        for part in HYPHEN_SPLIT_RE.split(tok):
            part = part.strip()
            if not part:
                continue
            for ch in part:
                if ch.isalpha() or ch.isdigit():
                    letters.append(ch.upper())
                    break
    return "".join(letters)


def _acr_key(acr_norm: str) -> str:
    """
    Canonicalise an acronym for matching by stripping non-alphanumerics and uppercasing.
    Keeps digits and letters only to ensure stable comparisons across punctuation variants.

    Args:
        acr_norm: Acronym surface/normalised form.

    Returns:
        Uppercased alphanumeric-only acronym key.
    """
    return "".join(ch for ch in acr_norm if ch.isalnum()).upper()


def _initials_match_backref(acr_norm: str, clean: str) -> bool:
    """
    Check whether a cleaned candidate’s initials match the acronym key.
    Uses a hyphen-aware initials builder to handle cases like "Single sign-on" → "SSO".

    Args:
        acr_norm: Acronym to match (may include punctuation/casing).
        clean: Cleaned candidate definition text.

    Returns:
        True if initials(clean) == canonical(acr_norm), else False.
    """
    a = _acr_key(acr_norm)
    return _initials_hyphen_aware(clean) == a


CONF_MAX = 0.99


def clamp_confidence(x: float, *, cap: float = CONF_MAX) -> float:
    """
    Clamp a confidence score into [0.0, cap] to avoid returning a hard 1.0.

    Args:
        x: Candidate confidence score.
        cap: Upper bound to clamp to (default 0.99).

    Returns:
        Clamped confidence value.
    """
    return 0.0 if x < 0.0 else (cap if x > cap else x)


def _titlecase_ratio(s: str) -> float:
    toks = TOKEN_RE.findall(s)
    if not toks:
        return 0.0
    good = 0
    for t in toks:
        # "USA" counts as good; "Sign-on" counts if 'S' etc.
        if t and (t[0].isupper() or t.isupper()):
            good += 1
    return good / len(toks)


def _valid_backref_candidate(
    *,
    clean: str,
    acr_norm: str,
    max_chars: int,
    require_two_words: bool,
) -> bool:
    if not clean:
        return False

    # enforce max_chars consistently (applies to both definitionish + initials)
    if len(clean) > max_chars:
        return False

    # reject “candidate is the acronym”
    if clean.replace(" ", "").upper() == acr_norm.replace(" ", ""):
        return False

    if require_two_words and len(TOKEN_RE.findall(clean)) < 2:
        return False

    # critical: prevent LOL being accepted for LLO (your failing test)
    # but allow hyphen-aware initials so "sign-on" yields S + O
    if not (initials_match(acr_norm, clean) or _initials_match_backref(acr_norm, clean)):
        return False

    return True


def _score_backref_confidence(
    *,
    cfg: ExtractionConfig,
    fo_surface: str,
    cand: str,
    evidence: BackrefEvidence,
    back: int,  # 1 = previous sentence, 2 = two sentences back, etc.
    dist_chars: int,  # char distance from prev sentence end to FO start
) -> tuple[float, tuple[str, ...]]:
    conf_cfg = getattr(cfg, "confidence", None)

    base = base_conf_for(cfg, source="sentence_backref")

    # knobs (with safe fallbacks)
    def _get(name: str, default: float) -> float:
        return getattr(conf_cfg, name, default) if conf_cfg is not None else default

    rs: list[str] = []

    score = base
    rs.append(f"base={base:.4f}")

    if evidence == "definitionish":
        boost = _get("backref_definitionish_boost", 0.10)
        score += boost
        rs.append(f"evidence=definitionish:+{boost:.4f}")
    else:
        boost = _get("backref_initials_boost", 0.00)
        score += boost
        rs.append(f"evidence=initials:+{boost:.4f}")

    lb_pen = max(0, back - 1) * _get("backref_lookback_penalty", 0.05)
    if lb_pen:
        score -= lb_pen
        rs.append(f"lookback={back}:-{lb_pen:.4f}")

    cap = int(getattr(conf_cfg, "backref_distance_penalty_cap_chars", 200)) if conf_cfg is not None else 200
    per = _get("backref_distance_penalty_per_char", 0.0005)
    dist_eff = min(max(dist_chars, 0), cap)
    dist_pen = dist_eff * per
    if dist_pen:
        score -= dist_pen
        rs.append(f"dist_chars={dist_eff}:-{dist_pen:.4f}")

    if fo_surface.isupper():
        b = _get("backref_uppercase_acronym_boost", 0.05)
        score += b
        rs.append(f"acr_caps:+{b:.4f}")

    ratio = _titlecase_ratio(cand)
    thr = _get("backref_titlecase_ratio_threshold", 0.80)
    if ratio >= thr:
        b = _get("backref_titlecase_boost", 0.05)
        score += b
        rs.append(f"titlecase={ratio:.2f}:+{b:.4f}")
    else:
        rs.append(f"titlecase={ratio:.2f}:0")

    score = clamp_confidence(score)
    rs.append(f"final={score:.4f}")

    return score, tuple(rs)


def _candidate_from_prev_sentence(
    *,
    acr_norm: str,
    prev_text: str,
    cfg: ExtractionConfig,
    max_chars: int,
    require_two_words: bool,
) -> tuple[str, BackrefEvidence] | None:
    """Extract a defensible long-form candidate from the previous sentence.

    This helper tries to recover a “definition-ish” phrase from the prior sentence
    and validate it using the shared `clean_definition` pipeline.

    Strategy (in order):
      1) Tighten the previous sentence down to a plausible title/definition span
         via `tighten_definition_span`, then clean/validate it.
      2) If that fails, fall back to an initials-based shortest span
         (`best_span_by_initials`), then clean/validate it.

    The returned `BackrefEvidence` indicates which route produced the candidate
    (e.g., `BackrefEvidence.DEFINITIONISH` vs `BackrefEvidence.INITIALS`).

    Args:
        acr_norm: Normalised acronym (typically uppercased) to match against.
        prev_text: Raw previous-sentence slice from the document.
        cfg: Extraction configuration used by `clean_definition` and related guardrails.
        max_chars: Maximum allowed candidate length (characters) for span selection.
        require_two_words: If True, require the cleaned candidate to contain >=2 tokens.

    Returns:
        (candidate, evidence) if a cleaned candidate passes guardrails; otherwise None.
    """
    prev_raw = prev_text.strip()
    if not prev_raw:
        return None

    sent = prev_raw.rstrip(" \t\r\n.?!…;:")

    # 1) definition-ish path
    base = tighten_definition_span(sent)
    clean = clean_definition(base, acr_norm=acr_norm, cfg=cfg, kind="inline")

    if clean and _valid_backref_candidate(
        clean=clean, acr_norm=acr_norm, max_chars=max_chars, require_two_words=require_two_words
    ):
        return clean, "definitionish"
    # else: fall through

    # 2) initials fallback
    cand = best_span_by_initials(acr_norm, sent, max_chars=max_chars)
    if not cand:
        return None

    clean = clean_definition(cand, acr_norm=acr_norm, cfg=cfg, kind="inline")

    if clean and _valid_backref_candidate(
        clean=clean, acr_norm=acr_norm, max_chars=max_chars, require_two_words=require_two_words
    ):
        return clean, "initials"

    return None


def _find_backref_candidate(
    *,
    text: str,
    spans: list[Span],
    si: int,
    acr_norm: str,
    cfg: ExtractionConfig,
    max_chars: int,
    require_two_words: bool,
) -> tuple[str, Span, int, BackrefEvidence] | None:
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

        hit = _candidate_from_prev_sentence(
            acr_norm=acr_norm,
            prev_text=prev_slice,
            cfg=cfg,
            max_chars=max_chars,
            require_two_words=require_two_words,
        )
        if hit:
            cand, evidence = hit
            return cand, (prev_s, prev_e), back, evidence

    return None


def _emit_backref_def(
    *,
    acr_norm: str,
    fo: FirstOccurrence,
    cand: str,
    prev_span: Span,
    text: str,
    cfg: ExtractionConfig,
    back: int,
    evidence: BackrefEvidence,
) -> ExtractedDefinition:
    prev_s, prev_e = prev_span
    dist_chars = max(0, fo.start_offset - prev_e)

    conf, reasons = _score_backref_confidence(
        cfg=cfg,
        fo_surface=fo.acronym,
        cand=cand,
        evidence=evidence,
        back=back,
        dist_chars=dist_chars,
    )

    return ExtractedDefinition(
        acronym=acr_norm,
        definition=cand,
        source="sentence_backref",
        definition_confidence=conf,
        acr_start=fo.start_offset,
        acr_end=fo.end_offset,
        def_start=prev_s,
        def_end=prev_e,
        original_definition=text[prev_s:prev_e].strip(),
        kind="sentence_backref",
        reasons=reasons,
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

        cand, prev_span, back, evidence = hit
        out.append(
            _emit_backref_def(
                acr_norm=acr_norm,
                fo=fo,
                cand=cand,
                prev_span=prev_span,
                text=text,
                cfg=cfg,
                back=back,
                evidence=evidence,
            )
        )

    return out
