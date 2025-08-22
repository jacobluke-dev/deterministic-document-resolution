from typing import Awaitable, Callable, Iterable, Optional, Protocol, TypeAlias, cast, runtime_checkable

from plainera_core.domain import DefinitionCandidate
from plainera_core.services.resolver import AcronymResolver


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

"""Type alias for a synchronous lookup function.

Args:
    str: The acronym text to resolve.

Returns:
    Iterable[DefinitionCandidateLike]: A collection of candidate definitions.
"""
LookupFunc: TypeAlias = Callable[[str], Iterable[DefinitionCandidateLike]]
"""Type alias for a resolver return type.

The resolver may be synchronous or asynchronous.

Returns:
    Iterable[DefinitionCandidateLike] | Awaitable[Iterable[DefinitionCandidateLike]]:
        Candidate definitions either directly or wrapped in an awaitable.
"""
ResolveReturn: TypeAlias = Iterable[DefinitionCandidateLike] | Awaitable[Iterable[DefinitionCandidateLike]]

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
        return [DefinitionCandidate(text=acronym_text, score=0.5)]
    return _lookup


def create_resolver(lookup: Optional[LookupFunc] = None) -> AcronymResolverLike:
    """Factory for constructing the acronym resolver.

    This is the single entry point for creating an `AcronymResolver`
    with its collaborators. If no lookup function is provided, it falls
    back to the stub from `default_lookup()`.

    Args:
        lookup (Optional[LookupFunc], optional): A function to resolve
            acronym text into candidates. If None, the default stub is used.

    Returns:
        AcronymResolverLike: A resolver conforming to the minimal protocol,
        safe for injection into the API layer.
    """
    return cast(AcronymResolverLike, AcronymResolver(lookup or default_lookup()))
