from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # only used by type checkers; mypy already sees services/... via PYTHONPATH in Makefile
    from public_api.types import DefinitionCandidateLike
else:
    # runtime-compatible fallback so we don't need FastAPI or the service package installed
    DefinitionCandidateLike = Mapping[str, Any]

from document_resolution_core.core.domain import Acronym


class AcronymResolver:
    def __init__(self, lookup: Callable[[str], Iterable[DefinitionCandidateLike]]) -> None:
        self._lookup = lookup

    def resolve(self, acro: Acronym, top_k: int = 5):
        items = sorted(self._lookup(acro.text), key=lambda c: c.score, reverse=True)
        return items[:top_k]
