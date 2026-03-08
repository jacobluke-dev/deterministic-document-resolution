from plainera_unacronym.nlp.common.shared import normalize_acronym_key, strip_trailing_punct_str
from plainera_unacronym.nlp.common.types import AcronymDetectorConfig, Occurrence, OccurrenceBuildError
from plainera_unacronym.nlp.detection.heuristics.core import context_window, reason_tags
from plainera_unacronym.nlp.detection.heuristics.general import strip_terminal_plural


def adjust_end_for_trailing_dot(cfg: AcronymDetectorConfig, text: str, s: int, e: int) -> int:
    """
    Apply the dotted-display policy to an occurrence end-offset.

    If `cfg.dotted_display == "preserve"` and the character immediately following the
    matched span (`text[e]`) is a literal '.', advance the end offset by one so the
    occurrence span includes that trailing dot (e.g. matching "U.S" in "U.S." yields an
    end offset that includes the final period). In "strip" mode (or when no trailing dot
    exists), the end offset is returned unchanged.

    This function validates that the resulting span `[s, end_for_occ)` is a well-formed
    slice into `text`.

    Args:
        cfg: Detection configuration; reads `dotted_display` ("strip" or "preserve").
        text: The full source text the offsets refer to.
        s: Start offset (inclusive) of the matched surface.
        e: End offset (exclusive) of the matched surface (before trailing-dot adjustment).

    Returns:
        int: The adjusted end offset (exclusive) to use for the occurrence.

    Raises:
        OccurrenceBuildError: If the adjusted offsets are invalid (out of bounds or
            start/end ordering is wrong).
    """
    display_mode = getattr(cfg, "dotted_display", "strip")
    has_trailing_dot = e < len(text) and text[e] == "."

    # Surface to display (may include the trailing dot if preserving)
    end_for_occ = e + 1 if (display_mode == "preserve" and has_trailing_dot) else e

    if not (0 <= s < end_for_occ <= len(text)):
        raise OccurrenceBuildError("bad_offsets")

    return end_for_occ


def normalize_surface_for_key(surface: str) -> str:
    """
    Normalise a matched surface into (base_surface, key_base) for occurrence/key construction.

    The `base_surface` is produced by stripping terminal plural suffixes from fully-uppercase
    acronym tokens (e.g. "GPUs" -> "GPU", "CPU's" -> "CPU"). The `key_base` is then derived
    from `base_surface` by removing trailing punctuation via `strip_trailing_punct_str()`,
    ensuring acronym/key strings do not end with punctuation.

    Note:
        This does not canonicalize internal punctuation (e.g. dotted initialisms) — that is
        handled later by `normalize_acronym_key(..., dotted_mode=cfg.dotted_display)`.

    Args:
        surface: Raw matched surface form, typically `text[s:e]`.

    Returns:
        str:
            - key_base: `base_surface` with trailing punctuation stripped, suitable for
              key normalization and for storing as the occurrence acronym.
    """
    # IMPORTANT: strip trailing punct from base so acronym/key never has terminal dot
    return strip_trailing_punct_str(strip_terminal_plural(surface))


def build_occurrence_from_match(
    cfg: AcronymDetectorConfig,
    text: str,
    surface: str,
    s: int,
    e: int,
    conf: float,
) -> tuple[Occurrence, str]:
    """
    Build an `Occurrence` from a single candidate span and return it with its display key.

    Applies the dotted-display policy from `cfg.dotted_display` ("strip" or "preserve").
    preserve mode extends the occurrence span to include a trailing dot (offset only);
    the acronym/key do not retain terminal punctuation
    When `cfg.debug_reasons` is enabled, reason tags are attached.

    Args:
        cfg: Detection configuration
        text: Full source text.
        surface: Matched surface form (typically `text[s:e]`).
        s: Start offset (inclusive) of the match.
        e: End offset (exclusive) of the match before any trailing-dot adjustment.
        conf: Confidence score for this match.

    Returns:
        tuple[Occurrence, str]: The constructed `Occurrence` and its normalized
        display key (used for deduping/first-occurrence tracking).

    Raises:
        OccurrenceBuildError: If occurrence is invalid, not of type `str`, or empty, or poor
        offsets.

    Notes:
        * Trailing-dot handling is offset-based only; no regex is modified.
        * `context_window` is derived from the adjusted `(s, end_for_occ)` span.
        * Normalization uses `normalize_key(..., dotted_mode=cfg.dotted_display)`.
    """
    end_for_occ = adjust_end_for_trailing_dot(cfg, text, s, e)

    key_base = normalize_surface_for_key(surface)
    if not key_base.strip():
        raise OccurrenceBuildError("empty_acronym")

    display_key = normalize_acronym_key(
        key_base,
        cfg.allow_chars,
        dotted_mode=cfg.dotted_display,   # this governs INTERNAL dot handling (U.S.A vs USA)
    )
    if not display_key:
        raise OccurrenceBuildError("empty_display_key")

    ctx = context_window(text, s, end_for_occ, cfg.window_chars)

    # Optional reason tags; keep the same in serial + parallel
    rsn = tuple(reason_tags(surface, text, s, end_for_occ, cfg)) if getattr(cfg, "debug_reasons", False) else None

    occ = Occurrence(
        acronym=key_base,
        start_offset=s,
        end_offset=end_for_occ,
        occurrence_confidence=conf,
        segment_window=ctx,
        normalized_key=display_key,
        reasons=rsn,
    )
    return occ, display_key
