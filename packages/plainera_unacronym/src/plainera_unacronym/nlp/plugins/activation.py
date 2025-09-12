# plainera_unacronym/nlp/plugins/activation.py
from typing import FrozenSet
from .registry import DOMAIN_PLUGINS
from .interface import SupportsSniff
from ..types import DetectorConfig


def autodetect_domains(text: str, cfg: DetectorConfig) -> FrozenSet[str]:
    # If caller explicitly set domains, respect that (no auto).
    if cfg.enabled_domains:
        return cfg.enabled_domains
    detected = {name for name, plug in DOMAIN_PLUGINS.items()
                if isinstance(plug, SupportsSniff) and plug.sniff(text)}
    return frozenset(detected)
