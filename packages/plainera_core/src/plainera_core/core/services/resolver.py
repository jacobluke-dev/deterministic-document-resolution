from typing import Callable, Iterable

from public_api.types import DefinitionCandidateLike

from plainera_core.core.domain import Acronym


class AcronymResolver:
    def __init__(self, lookup: Callable[[str], Iterable[DefinitionCandidateLike]]) -> None:
        self._lookup = lookup

    def resolve(self, acro: Acronym, top_k: int = 5):
        items = sorted(self._lookup(acro.text), key=lambda c: c.score, reverse=True)
        return items[:top_k]
