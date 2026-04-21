from __future__ import annotations

from document_resolution.nlp.common.types import AcronymDetectorConfig, DefinedTermDetectorConfig

from .interface import SupportsSniff
from .registry import DOMAIN_PLUGINS


def autodetect_domains(
    text: str, _cfg: AcronymDetectorConfig | DefinedTermDetectorConfig, *, cap: int = 80_000
) -> frozenset[str]:
    """Return the set of domain plugin names auto-detected for a given text.
    This scans a capped prefix of ``text`` (default: first 80,000 characters)


    Args:
      text: Source document to inspect.
      _cfg: Current detector configuration. Accepted for interface symmetry and
        possible future use; not read by this implementation.
      cap: Maximum number of leading characters of ``text`` to inspect for
        speed and safety on very large inputs.

    Returns:
      frozenset[str]: Names of plugins (as registered in ``DOMAIN_PLUGINS``)
      whose ``sniff(text)`` returned True.

    """
    t = text[:cap]
    detected = {
        name for name, plug in DOMAIN_PLUGINS.items() if isinstance(plug, SupportsSniff) and _safe_sniff(plug, t)
    }
    return frozenset(detected)


def _safe_sniff(plug: SupportsSniff, text: str) -> bool:
    """Run a plugin's ``sniff`` method defensively and normalize the result.

    Args:
      plug: A plugin instance implementing ``sniff(text) -> bool``.
      text: The (possibly truncated) text to inspect.

    Returns:
      True if the plugin reports a match; otherwise False. Returns False on
      exceptions as well.
    """
    try:
        return bool(plug.sniff(text))
    except Exception:
        return False
