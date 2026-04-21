from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from document_resolution.nlp.common.types import TextSpanTuple


class DomainPlugin(Protocol):
    """Interface for domain-specific detection hooks.

    A DomainPlugin can:
      * contribute **extra candidates** the generic pattern might miss; and
      * **rescue/keep** borderline generic candidates based on domain context.

    Attributes:
        name: Stable domain name (e.g. "bio", "finance"). Used for registration and
            to check if the domain is enabled via `cfg.enabled_domains`.
    """

    name: str

    def extra_candidates(self, text: str, cfg) -> Iterator[TextSpanTuple]: ...

    def keep_guard(self, surface: str, text: str, s: int, e: int, cfg) -> bool: ...


@runtime_checkable
class SupportsSniff(Protocol):
    """Optional capability for **auto-enabling** a domain.

    Implement on a plugin if it can quickly determine whether a document likely
    belongs to its domain (used by `autodetect_domains`).
    """

    def sniff(self, text: str) -> bool: ...
