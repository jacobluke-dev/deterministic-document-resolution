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

from dataclasses import dataclass
from typing import Literal, Mapping

from plainera_unacronym.nlp import FirstOccurrence
from plainera_unacronym.nlp.common.constants_regex import HYPHEN_SPLIT_RE, TOKEN_RE
from plainera_unacronym.nlp.common.types import ExtractedDefinition, Span
from plainera_unacronym.nlp.extraction.anchored.clean import clean_definition
from plainera_unacronym.nlp.extraction.anchored.normalise import tighten_definition_span
from plainera_unacronym.nlp.extraction.backref.spans import best_span_by_initials, find_span_index, sent_spans
from plainera_unacronym.nlp.extraction.config import ExtractionConfig
from plainera_unacronym.nlp.extraction.core.collect import initials_match
from plainera_unacronym.nlp.extraction.engine.confidence import base_conf_for, conf_knob

BackrefEvidence = Literal["definitionish", "initials"]
CONF_MAX = 0.99


@dataclass(frozen=True, slots=True)
class _ScoreCtx:
    cfg: ExtractionConfig
    fo_surface: str
    cand: str
    evidence: BackrefEvidence
    back: int
    dist_chars: int

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
    """Return the fraction of tokens that look title-cased.

    A token counts as "title-ish" if:
      - its first character is uppercase (e.g., "Single"), OR
      - the whole token is uppercase (e.g., "USA").

    Used as a weak heuristic that a candidate definition resembles a proper noun / label.

    Args:
        s: Candidate definition text.

    Returns:
        Ratio in [0.0, 1.0]. Returns 0.0 when no tokens are found.
    """
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
    """
    Validate a cleaned back-reference candidate definition against hard guardrails.
    Enforces length, non-identity with the acronym, optional token-count minimum,
    and an initials match check (strict or hyphen-aware fallback).
    Returns True only when the candidate is structurally defensible.

    Args:
        clean: Candidate definition after cleaning/normalisation.
        acr_norm: Normalised acronym key used for comparison/matching.
        max_chars: Maximum permitted character length for the candidate.
        require_two_words: Whether the candidate must contain at least two tokens.

    Returns:
        True if the candidate passes all guardrails, otherwise False.
    """
    if not clean:
        return False

    if len(clean) > max_chars:
        return False

    clean_comp = clean.replace(" ", "").upper()
    acr_comp = acr_norm.replace(" ", "")

    if clean_comp == acr_comp:
        return False

    if require_two_words and len(TOKEN_RE.findall(clean)) < 2:
        return False

    return initials_match(acr_norm, clean) or _initials_match_backref(acr_norm, clean)



def _add_term(
    score: float,
    rs: list[str],
    *,
    label: str,
    delta: float,
) -> float:
    """Apply a signed delta to the score and append a trace line to `rs`.

    Records either:
      - "<label>:+<abs(delta)>" / "<label>:-<abs(delta)>" when delta != 0
      - "<label>:0" when delta == 0

    This keeps confidence scoring explainable and testable by preserving the
    exact additive terms used.

    Args:
        score: Current running score.
        rs: Reasons list to append to (mutated in place).
        label: Prefix label for the trace line (e.g., "lookback=2").
        delta: Signed amount to add to `score`.

    Returns:
        Updated score after applying `delta`.
    """
    if delta:
        sign = "+" if delta > 0 else "-"
        rs.append(f"{label}:{sign}{abs(delta):.4f}")
        return score + delta
    rs.append(f"{label}:0")
    return score


def _evidence_delta(ctx: _ScoreCtx) -> tuple[float, str]:
    """Return the additive confidence delta for the evidence type.

    Evidence types are:
      - "definitionish": candidate came from span tightening / definition-like selection
      - "initials": candidate came from initials-based span selection

    The returned label is formatted for the reasons trace (e.g., "evidence=initials").

    Args:
        ctx: Scoring context.

    Returns:
        (delta, label) where delta is added to the running score.
    """
    if ctx.evidence == "definitionish":
        v = conf_knob(ctx.cfg, "backref_definitionish_boost", 0.10)
        return v, "evidence=definitionish"
    v = conf_knob(ctx.cfg, "backref_initials_boost", 0.00)
    return v, "evidence=initials"


def _lookback_delta(ctx: _ScoreCtx) -> tuple[float, str]:
    """Return the lookback penalty based on how many sentences back the candidate is.

    Penalises candidates found further back than the immediately previous sentence:
        penalty = max(0, back - 1) * backref_lookback_penalty

    Args:
        ctx: Scoring context.

    Returns:
        (delta, label) where delta is negative (a penalty) and label is "lookback=<n>".
    """
    pen_per = conf_knob(ctx.cfg, "backref_lookback_penalty", 0.05)
    pen = max(0, ctx.back - 1) * pen_per
    return -pen, f"lookback={ctx.back}"


def _distance_delta(ctx: _ScoreCtx) -> tuple[float, str]:
    """Return the distance penalty based on character gap between candidate and FO.

    Uses `dist_chars` (distance from previous sentence end to FO start) and applies:
        penalty = min(max(dist_chars, 0), cap) * per_char

    Args:
        ctx: Scoring context.

    Returns:
        (delta, label) where delta is negative (a penalty) and label is "dist_chars=<n>".
    """
    cap = int(getattr(ctx.cfg, "backref_distance_penalty_cap_chars", 200)) if ctx.cfg is not None else 200
    per = conf_knob(ctx.cfg, "backref_distance_penalty_per_char", 0.0005)
    dist_eff = min(max(ctx.dist_chars, 0), cap)
    pen = dist_eff * per
    return -pen, f"dist_chars={dist_eff}"


def _acronym_caps_delta(ctx: _ScoreCtx) -> tuple[float, str]:
    """Return the uppercase-acronym boost if the FO surface is all-caps.

    This favours canonical acronym renderings (e.g., "SSO" vs "sso") as they tend to
    correlate with more intentional definitions in formal prose.

    Args:
        ctx: Scoring context.

    Returns:
        (delta, label) where delta is 0.0 if FO is not all-caps, else a positive boost.
    """

    if not ctx.fo_surface.isupper():
        return 0.0, "acr_caps"
    b = conf_knob(ctx.cfg, "backref_uppercase_acronym_boost", 0.05)
    return b, "acr_caps"


def _titlecase_delta(ctx: _ScoreCtx) -> tuple[float, str]:
    """Return the title-case boost if the candidate looks like a proper label.

    Computes `_titlecase_ratio(cand)` and compares to the configured threshold.
    If the ratio meets/exceeds the threshold, applies `backref_titlecase_boost`.

    Args:
        ctx: Scoring context.

    Returns:
        (delta, label) where delta is 0.0 when below threshold, else a positive boost.
        Label is "titlecase=<ratio>" for traceability.
    """
    ratio = _titlecase_ratio(ctx.cand)
    thr = conf_knob(ctx.cfg, "backref_titlecase_ratio_threshold", 0.80)
    if ratio < thr:
        return 0.0, f"titlecase={ratio:.2f}"
    b = conf_knob(ctx.cfg, "backref_titlecase_boost", 0.05)
    return b, f"titlecase={ratio:.2f}"


def _score_backref_confidence(
    *,
    cfg: ExtractionConfig,
    fo_surface: str,
    cand: str,
    evidence: BackrefEvidence,
    back: int,
    dist_chars: int,
) -> tuple[float, tuple[str, ...]]:
    """Score backref confidence using deterministic additive terms.

    Args:
        cfg: Extraction config (reads cfg.confidence knobs).
        fo_surface: Acronym as it appears at FO (used for caps heuristic).
        cand: Cleaned candidate definition.
        evidence: Candidate source route (definitionish vs initials).
        back: Sentence lookback count (1 = nearest previous).
        dist_chars: Char distance from prev sentence end to FO start.

    Returns:
        (score, reasons) where score is clamped to [0.0, 0.99].
    """
    base = base_conf_for(cfg, source="sentence_backref")

    ctx = _ScoreCtx(
        cfg=cfg,
        fo_surface=fo_surface,
        cand=cand,
        evidence=evidence,
        back=back,
        dist_chars=dist_chars,
    )

    rs: list[str] = [f"base={base:.4f}"]
    score = base

    for fn in (_evidence_delta, _lookback_delta, _distance_delta, _acronym_caps_delta, _titlecase_delta):
        delta, label = fn(ctx)
        score = _add_term(score, rs, label=label, delta=delta)

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
    """
    Build an `ExtractedDefinition` for a sentence back-reference hit.
    Computes char-distance from the previous sentence end to the FO start and
    scores confidence/reasons via `_score_backref_confidence`, then maps spans
    back to absolute offsets and stores the raw prior-sentence slice.

    Args:
        acr_norm: Normalised acronym key to store on the definition.
        fo: First-occurrence metadata for the acronym (absolute offsets).
        cand: Cleaned candidate long-form definition.
        prev_span: Absolute (start, end) span of the previous sentence in `text`.
        text: Full document text.
        cfg: Extraction configuration (confidence knobs, limits).
        back: Lookback distance in sentences (1 = nearest previous sentence).
        evidence: Which extraction route produced `cand` ("definitionish" vs "initials").

    Returns:
        ExtractedDefinition: A fully-populated backref definition record.
    """
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
