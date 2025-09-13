# plainera_unacronym/nlp/plugins/activation.py
from typing import FrozenSet
from .registry import DOMAIN_PLUGINS
from .interface import SupportsSniff
from ..types import DetectorConfig


def autodetect_domains(text: str, cfg: DetectorConfig, *, cap: int = 80_000) -> FrozenSet[str]:
    # Keep this pure: detect only. Caller decides how to merge.
    t = text[:cap]
    detected = {
        name for name, plug in DOMAIN_PLUGINS.items()
        if isinstance(plug, SupportsSniff) and _safe_sniff(plug, t)
    }
    return frozenset(detected)

def _safe_sniff(plug: SupportsSniff, text: str) -> bool:
    try:
        return bool(plug.sniff(text))
    except Exception:
        return False
