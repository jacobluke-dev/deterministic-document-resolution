from plainera_core.core.domain import Acronym


class AcronymResolver:
    def __init__(self, lookup):
        self._lookup = lookup  # callable(str) -> list[DefinitionCandidate]

    def resolve(self, acro: Acronym, top_k: int = 5):
        items = sorted(self._lookup(acro.text), key=lambda c: c.score, reverse=True)
        return items[:top_k]
