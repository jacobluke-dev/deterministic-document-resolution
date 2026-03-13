from __future__ import annotations

from typing import FrozenSet

from plainera_unacronym.nlp.common.types import AcronymDetectorConfig, DefinedTermDetectorConfig

from .interface import SupportsSniff
from .registry import DOMAIN_PLUGINS


def autodetect_domains(text: str,
                       _cfg: AcronymDetectorConfig| DefinedTermDetectorConfig,
                       *,
                       cap: int = 80_000) -> FrozenSet[str]:
    """Return the set of domain plugin names auto-detected for a given text.

    This scans a capped prefix of ``text`` (default: first 80,000 characters)
    and asks each registered plugin that implements ``SupportsSniff`` whether
    the document appears to match its domain. Misbehaving plugins are sandboxed
    via ``_safe_sniff`` so a single failure does not break detection.

    The function is **agnostic**: it only detects and returns names. The caller
    is responsible for merging the result into ``cfg.enabled_domains`` and
    (optionally) supplying per-domain configs via ``cfg.domain_cfg``.

    Args:
      text: Source document to inspect.
      _cfg: Current detector configuration. Accepted for interface symmetry and
        possible future use; not read by this implementation.
      cap: Maximum number of leading characters of ``text`` to inspect for
        speed and safety on very large inputs.

    Returns:
      FrozenSet[str]: Names of plugins (as registered in ``DOMAIN_PLUGINS``)
      whose ``sniff(text)`` returned True.

    Notes:
      - Only plugins marked with the runtime-checkable ``SupportsSniff`` protocol
        are queried.
      - This function does not modify ``cfg``; it is up to the caller to merge
        the returned set with any pre-enabled domains. (this is done in Detector._with_auto_domains(..))

    Example:
      >>> auto = autodetect_domains(doc_text, cfg)
      >>> if auto:
      ...     cfg = dataclasses.replace(cfg, enabled_domains=cfg.enabled_domains | auto)
    """
    t = text[:cap]
    detected = {
        name for name, plug in DOMAIN_PLUGINS.items() if isinstance(plug, SupportsSniff) and _safe_sniff(plug, t)
    }
    return frozenset(detected)


def _safe_sniff(plug: SupportsSniff, text: str) -> bool:
    """Run a plugin's ``sniff`` method defensively and normalize the result.

    Invokes ``plug.sniff(text)`` and coerces the return value to ``bool``.
    Any exception raised by the plugin is caught and treated as a non-match
    (``False``). This ensures one misbehaving plugin cannot break domain
    auto-detection for the whole pipeline.

    Args:
      plug: A plugin instance implementing ``sniff(text) -> bool``.
      text: The (possibly truncated) text to inspect.

    Returns:
      True if the plugin reports a match; otherwise False. Returns False on
      exceptions as well.

    Notes:
      - Exceptions are intentionally swallowed to isolate failures.
      - Use a capped slice of the document upstream if performance matters.

    Example:
      >>> for name, plug in DOMAIN_PLUGINS.items():
      ...     if isinstance(plug, SupportsSniff) and _safe_sniff(plug, doc[:80000]):
      ...         detected.add(name)
    """
    try:
        return bool(plug.sniff(text))
    except Exception:
        return False
