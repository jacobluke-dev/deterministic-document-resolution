from typing import Callable, Iterable, Protocol, runtime_checkable

from plainera_core.core.domain import DefinitionCandidate

from public_api.types import ResolveReturn


class DefinitionCandidateLike(Protocol):
    """Structural protocol for definition candidates.
    """
    @property
    def text(self) -> str: ...

    @property
    def score(self) -> float: ...

class AcronymLike(Protocol):
    """Structural protocol for acronyms.
    """
    @property
    def text(self) -> str: ...


@runtime_checkable
class AcronymResolverLike(Protocol):
    """
    Protocol for objects that can resolve acronyms into definitions.
    """

    def resolve(self, acro: AcronymLike, top_k: int = 5) -> ResolveReturn:
        """Resolve an acronym to candidate definitions.
           Args:
               acro (AcronymLike): The acronym object containing text.
               top_k (int, optional): The maximum number of candidates to return.
                   Defaults to 5.

           Returns:
               ResolveReturn: Either a direct iterable of candidates or an awaitable
               producing candidates.
        """
        ...


# TODO UN-14 2.5 Story
def default_lookup() -> Callable[[str], Iterable[DefinitionCandidate]]:
    """Create a minimal stub lookup function.

    //TODO This is a placeholder implementation until Story 2.5 introduces
    a glossary or database-backed lookup.

    Returns:
        LookupFunc: A function that accepts acronym text and returns a
        list containing a single dummy candidate (with the acronym text
        itself and a neutral score).
    """
    def _lookup(acronym_text: str) -> Iterable[DefinitionCandidate]:
        # tuple avoids list invariance issues
        return (DefinitionCandidate(text=acronym_text, score=0.5),)
    return _lookup
