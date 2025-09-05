# packages/plainera_core/src/plainera_core/core/services/factory.py

from typing import Optional, cast

from plainera_core.core.services.resolver import AcronymResolver

from public_api.core.providers import AcronymResolverLike, default_lookup
from public_api.types import LookupFunc


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
