from .interface import DomainPlugin

DOMAIN_PLUGINS: dict[str, DomainPlugin] = {}


def register_plugin(p: DomainPlugin) -> None:
    DOMAIN_PLUGINS[p.name] = p


def get(names) -> list[DomainPlugin]:
    """Return registered plugins for the requested names, in order.

    Unknown names are ignored (treat as "not installed").
    """
    out: list[DomainPlugin] = []
    for n in names or ():
        p = DOMAIN_PLUGINS.get(n)
        if p is not None:
            out.append(p)
    return out
