from typing import Protocol, Iterator, Tuple, runtime_checkable

Span = Tuple[str, int, int]

class DomainPlugin(Protocol):
    name: str
    def extra_candidates(self, text: str, cfg) -> Iterator[Span]: ...
    def keep_guard(self, surface: str, text: str, s: int, e: int, cfg) -> bool: ...


@runtime_checkable
class SupportsSniff(Protocol):
    def sniff(self, text: str) -> bool: ...
