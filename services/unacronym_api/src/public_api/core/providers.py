from typing import Protocol, runtime_checkable

from plainera_core.core.domain import DefinitionCandidate

from public_api.types import LookupFunc, ResolveReturn


class DefinitionCandidateLike(Protocol):
    """Structural protocol for definition candidates.

    Attributes:
        text (str): The human-readable definition text.
        score (float): A relevance/confidence score for the candidate.
    """
    text: str
    score: float

class AcronymLike(Protocol):
    """Structural protocol for acronyms.

    Attributes:
        text (str): The acronym text to be resolved.
    """
    text: str


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
def default_lookup() -> LookupFunc:
    """Create a minimal stub lookup function.

    //TODO This is a placeholder implementation until Story 2.5 introduces
    a glossary or database-backed lookup.

    Returns:
        LookupFunc: A function that accepts acronym text and returns a
        list containing a single dummy candidate (with the acronym text
        itself and a neutral score).
    """
    def _lookup(acronym_text: str) -> list[DefinitionCandidateLike]:
        return [DefinitionCandidate(text=acronym_text, score=0.5)]   # type: ignore[incompatible-type]
    return _lookup
