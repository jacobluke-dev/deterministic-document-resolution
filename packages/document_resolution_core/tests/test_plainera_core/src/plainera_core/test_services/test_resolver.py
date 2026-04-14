from document_resolution_core.core.domain import Acronym
from document_resolution_core.core.services.resolver import AcronymResolver


class TestAcronymResolver:

    def test_acronym_resolver_empty_lookup(self):
        """
        Test AcronymResolver handles empty lookup results gracefully.
        """
        resolver = AcronymResolver(lambda x: [])
        resolved = resolver.resolve(Acronym("XYZ"))
        assert resolved == []  # Expecting an empty list
