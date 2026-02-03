from typing import Iterator, Protocol, runtime_checkable

from plainera_unacronym.nlp.common.types import TextSpanTuple

# TODO move this somewhere globally
"""A candidate acronym span.

Elements:
    0 (str): Surface text as it appears in `text` (not normalized).
    1 (int): Start offset (inclusive) in `text`.
    2 (int): End offset (exclusive) in `text`.
"""


class DomainPlugin(Protocol):
    """Interface for domain-specific detection hooks.

    A DomainPlugin can:
      * contribute **extra candidates** the generic pattern might miss; and
      * **rescue/keep** borderline generic candidates based on domain context.

    Attributes:
        name: Stable domain name (e.g. "bio", "finance"). Used for registration and
            to check if the domain is enabled via `cfg.enabled_domains`.

    Methods:
        extra_candidates(text, cfg):
            Yield additional `(surface, start, end)` spans discovered with
            domain-specific rules (e.g., special regexes). Only called when the
            plugin’s `name` is enabled in `cfg.enabled_domains`.

            Args:
                text: Full source text.
                cfg: Active detector configuration.

            Yields:
                Span: `(surface, start, end)` with end-exclusive indices.

        keep_guard(surface, text, s, e, cfg):
            Decide whether to **keep** a generic candidate that would otherwise be
            dropped (e.g., short/ambiguous tokens). Use nearby context windows,
            sentence boundaries, or domain heuristics.

            Args:
                surface: Matched surface text (`text[s:e]`).
                text: Full source text.
                s: Start offset (inclusive).
                e: End offset (exclusive).
                cfg: Active detector configuration (may contain per-domain config
                    under `cfg.domain_cfg[plugin.name]`).

            Returns:
                bool: `True` to keep the candidate; `False` to let the generic
                pipeline decide (possibly dropping it).

    Notes:
        * Implementations should be fast and side-effect free.
        * If a plugin does not need rescues, `keep_guard` can simply return False.
    """

    name: str

    def extra_candidates(self, text: str, cfg) -> Iterator[TextSpanTuple]: ...

    def keep_guard(self, surface: str, text: str, s: int, e: int, cfg) -> bool: ...


@runtime_checkable
class SupportsSniff(Protocol):
    """Optional capability for **auto-enabling** a domain.

    Implement on a plugin if it can quickly determine whether a document likely
    belongs to its domain (used by `autodetect_domains`).

    Methods:
        sniff(text):
            Lightweight check (on a capped prefix of `text`) for domain cues.

            Args:
                text: Full source text (caller may pass a truncated prefix).

            Returns:
                bool: `True` if the domain should be **enabled** for this document.

    Notes:
        * Marked `@runtime_checkable` so callers can use `isinstance(plugin, SupportsSniff)`.
        * Keep this fast and robust; failures should not raise (callers may sandbox).
    """

    def sniff(self, text: str) -> bool: ...
