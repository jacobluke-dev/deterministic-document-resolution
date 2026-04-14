from typing import Optional

from document_resolution.nlp.extraction.acronyms.config import ConfidenceConfig, ExtractionConfig


def base_for_kind(cfg: ExtractionConfig, kind: str) -> float:
    """
    Map an anchored-pattern `kind` to the appropriate base confidence.

    Anchored extraction emits different `kind` labels depending on which regex
    pattern fired (wrapper/parenthetical vs inline cue). This helper collapses
    those labels into a small set of scoring “sources” and returns the configured
    base confidence for that source.

    The mapping is intentionally coarse:
      - Wrapper / parenthetical forms (e.g., "Long Form (ACR)", "ACR (Long Form)")
        map to `source="parenthetical"`.
      - Inline cue forms (e.g., "ACR stands for Long Form", "Long Form, abbreviated as ACR")
        map to `source="inline"`.
      - Everything else falls back to `source="first_occurrence_anchored"`.

    Args:
        cfg: Extraction configuration containing `cfg.confidence.base_by_source`.
        kind: Pattern kind label produced by anchored pattern specs (e.g., "def_before",
            "def_after_direct", "inline", "inline_before").

    Returns:
        The configured base confidence for the mapped source, or the default used by
        `base_conf_for` if the source is not configured.
    """
    # treat these as “wrapper”/parenthetical forms
    if kind in {
        "def_before",
        "def_before_direct",
        "def_after",
        "def_after_direct",
        "before_acr_paren",
        "paren_before_acr",
    }:
        return base_conf_for(cfg, source="parenthetical")
    # inline cue forms
    if kind in {"inline", "inline_before"}:
        return base_conf_for(cfg, source="inline")
    return base_conf_for(cfg, source="first_occurrence_anchored")  # fallback


def base_conf_for(cfg: ExtractionConfig, *, source: str, default: float = 0.50) -> float:
    """
    Fetch the configured base confidence for a given provenance/scoring `source`.

    This is the single lookup point for “base confidence” values used across
    extraction stages. It reads from `cfg.confidence.base_by_source` and returns
    a safe fallback if the key is missing.

    Args:
        cfg: Extraction configuration containing `cfg.confidence.base_by_source`.
        source: The provenance/scoring key to look up (e.g., "parenthetical", "inline",
            "backref", "first_occurrence_anchored").
        default: Value to return if `source` is not present in `base_by_source`.

    Returns:
        A float base confidence. If `source` is configured, returns that value;
        otherwise returns `default`.
    """
    v = cfg.confidence.base_by_source.get(source)
    return default if v is None else v


def conf_knob(cfg: ExtractionConfig, name: str, default: float) -> float:
    """Read a numeric confidence knob from cfg.confidence with a safe default.

    Intended for scalar tuning parameters (boosts/penalties/weights) that live on
    `ConfidenceConfig` (e.g. `backref_lookback_penalty`, `dist_weight`).

    Args:
        cfg: ExtractionConfig (may or may not have `confidence` set).
        name: Attribute name on `ConfidenceConfig`.
        default: Value to return if `cfg.confidence` is missing or attribute absent.

    Returns:
        The configured knob value or `default`.
    """
    cc: Optional[ConfidenceConfig] = getattr(cfg, "confidence", None)
    if cc is None:
        return default
    v = getattr(cc, name, None)
    return default if v is None else float(v)
