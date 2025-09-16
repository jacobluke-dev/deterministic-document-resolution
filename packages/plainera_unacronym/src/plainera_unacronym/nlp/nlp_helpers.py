import hashlib, json
import heapq
from typing import Any

from plainera_unacronym.nlp import FirstOccurrence, DetectorConfig


def _cfg_fingerprint(cfg: DetectorConfig) -> str:
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


def top_n_values(firsts: dict[str, "FirstOccurrence"], n: int = 5)-> list[dict[str, str | float]] | list[Any]:
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
    items = heapq.nlargest(n, firsts.items(), key=lambda kv: kv[1].confidence)
    return [{"key": k, "conf": round(fo.confidence, 3)} for k, fo in items]
