from plainera_core.domain import Acronym
from plainera_core.services.resolver import AcronymResolver


class TestAcronymResolver:

    def test_acronym_resolver_empty_lookup(self):
        """
        Test AcronymResolver handles empty lookup results gracefully.
        """
        resolver = AcronymResolver(lambda x: [])
        resolved = resolver.resolve(Acronym("XYZ"))
        assert resolved == []  # Expecting an empty list
