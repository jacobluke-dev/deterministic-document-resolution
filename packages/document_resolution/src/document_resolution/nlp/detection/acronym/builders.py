from document_resolution.nlp.common.shared import normalize_acronym_key, strip_trailing_punct_str
from document_resolution.nlp.common.types import AcronymDetectorConfig, Occurrence, OccurrenceBuildError
from document_resolution.nlp.detection.heuristics.core import context_window, reason_tags
from document_resolution.nlp.detection.heuristics.general import strip_terminal_plural


def adjust_end_for_trailing_dot(cfg: AcronymDetectorConfig, text: str, s: int, e: int) -> int:
    """Apply dotted-display policy to an occurrence end offset.

    Args:
        cfg: Detector configuration.
        text: Full source text.
        s: Match start offset.
        e: Match end offset before trailing-dot adjustment.

    Returns:
        Adjusted end offset.

    Raises:
        OccurrenceBuildError: If the resulting slice bounds are invalid.
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
    Normalise a matched surface for occurrence and key construction.

    Strips terminal plural suffixes and trailing punctuation, but leaves internal
    punctuation handling to later key normalisation.

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
    """Build an occurrence from a matched span and return it with its display key.

    Applies dotted-display offset handling, normalises the surface for acronym/key
    construction, derives the context window, and optionally attaches debug reasons.

    Args:
        cfg: Detector configuration.
        text: Full source text.
        surface: Matched surface form.
        s: Match start offset.
        e: Match end offset before trailing-dot adjustment.
        conf: Confidence score for the match.

    Returns:
        Constructed occurrence and its normalised display key.

    Raises:
        OccurrenceBuildError: If the occurrence surface, key, or offsets are invalid.
    """
    end_for_occ = adjust_end_for_trailing_dot(cfg, text, s, e)

    key_base = normalize_surface_for_key(surface)
    if not key_base.strip():
        raise OccurrenceBuildError("empty_acronym")

    display_key = normalize_acronym_key(
        key_base,
        cfg.allow_chars,
        dotted_mode=cfg.dotted_display,  # this governs INTERNAL dot handling (U.S.A vs USA)
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
