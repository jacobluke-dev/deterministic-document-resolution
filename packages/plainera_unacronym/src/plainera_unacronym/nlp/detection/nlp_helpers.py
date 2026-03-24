from __future__ import annotations

import hashlib
import heapq
import json
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from plainera_unacronym.nlp.common.types import FirstOccurrence


def cfg_fingerprint(cfg) -> str:
    """
    Compute a short, deterministic fingerprint of a detector configuration.

    Builds a JSON-serializable snapshot from selected fields and returns the
    first 12 hex characters of its SHA-256 hash. Useful for logging/comparison
    without emitting the full config.

    Fields included:
      - allow_chars
      - window_chars
      - min_confidence_default
      - dotted_display
      - enabled_domains (sorted)

    Args:
        cfg (DetectorConfig): An object exposing the attributes above (e.g., `DetectorConfig`).
             Missing attributes are treated as `None`.
    Returns:
        str: A 12-character lowercase hex fingerprint (stable for the same
        field values).

    Notes:
        * Intended for identification, not security; it’s a truncated hash.
        * Only the listed fields affect the fingerprint; changes to other
          config attributes are intentionally ignored.
    """
    data = {
        "allow_chars": getattr(cfg, "allow_chars", None),
        "window_chars": getattr(cfg, "window_chars", None),
        "min_conf": getattr(cfg, "min_confidence_default", None),
        "dotted_display": getattr(cfg, "dotted_display", None),
        "domains": sorted(getattr(cfg, "enabled_domains", []) or []),
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]


def _round_sig(x: float, sig: int = 3) -> float:
    """
    Round x to `sig` significant figures using Decimal(…, ROUND_HALF_UP).

    Examples:
        0.5555 @ 3sf → 0.556
        0.9999 @ 3sf → 1.0
        12345  @ 3sf → 12300
        0.00012345 @ 3sf → 0.000123
        -9.995 @ 3sf → -10.0
    Notes:
        * Converts via str() before Decimal to keep results predictable.
        * Returns a float for ease of use in normal code paths.
        * 0.0 is returned unchanged.
        * found round(x, k) wouldn't work i.e. round(0.5555,3) -> 0.555 WRONG!
    """
    d = Decimal(str(x))
    if d.is_zero():
        return 0.0
    exp = d.adjusted() - sig + 1
    q = Decimal(1).scaleb(exp)
    return float(d.quantize(q, rounding=ROUND_HALF_UP))


def top_n_values(firsts: dict[str, FirstOccurrence], n: int = 5) -> list[dict[str, str | float]] | list[Any]:
    """
    Return a compact preview of the top-N acronyms by confidence.

    Produces a list of small dicts with the normalized key and its confidence,
    sorted descending by confidence. No positions or context are included to
    keep logs lightweight and privacy-friendly.

    Args:
        firsts: Mapping from normalized key to `FirstOccurrence` (must have a
            `confidence` attribute).
        n: Maximum number of items to return. Defaults to 5.

    Returns:
         list[dict[str, str | float]] | list[Any]: Each element is
        `{"key": <normalized_key>, "conf": <confidence_rounded_to_3dp>}`.

    """
    if n <= 0 or not firsts:
        return []
    items = heapq.nlargest(n, firsts.items(), key=lambda kv: kv[1].occurrence_confidence)
    return [{"key": k, "conf": _round_sig(fo.occurrence_confidence, 3)} for k, fo in items]
