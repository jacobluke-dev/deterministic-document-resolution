from typing import Dict

from .interface import DomainPlugin

DOMAIN_PLUGINS: Dict[str, DomainPlugin] = {}


def register_plugin(p: DomainPlugin) -> None:
    DOMAIN_PLUGINS[p.name] = p
