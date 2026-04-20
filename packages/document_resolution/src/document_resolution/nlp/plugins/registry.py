from typing import Iterable

from .interface import DomainPlugin

DOMAIN_PLUGINS: dict[str, DomainPlugin] = {}


def register_plugin(p: DomainPlugin) -> None:
    """Register a domain plugin by its stable name.
    Args:
        p (DomainPlugin): Plugin instance to register.

    Returns:
        None
    """
    DOMAIN_PLUGINS[p.name] = p


def get(names: Iterable[str] | None) -> list[DomainPlugin]:
    """Return registered plugins for the requested names, preserving order.


    Args:
        names (Iterable[str] | None): Plugin names to resolve (e.g. cfg.enabled_domains).

    Returns:
        list[DomainPlugin]: Registered plugins corresponding to `names`, in order.
    """
    out: list[DomainPlugin] = []
    for n in names or ():
        p = DOMAIN_PLUGINS.get(n)
        if p is not None:
            out.append(p)
    return out
