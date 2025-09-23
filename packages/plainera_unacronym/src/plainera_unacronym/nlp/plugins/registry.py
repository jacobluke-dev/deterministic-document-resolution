from .interface import DomainPlugin

DOMAIN_PLUGINS: dict[str, DomainPlugin] = {}


def register_plugin(p: DomainPlugin) -> None:
    DOMAIN_PLUGINS[p.name] = p
