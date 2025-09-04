import pytest

from core.domain import Acronym


class TestAcronymFunctions:

    @pytest.fixture
    def acronym(self):
        """
        Fixture for creating Acronym objects.
        """
        return Acronym(text="API")

    def test_acronym_resolver_resolves_correctly(self, acronym_resolver, acronym):
        """
        Test AcronymResolver returns top-k definitions based on score.
        """
        resolved = acronym_resolver.resolve(acronym, top_k=2)
        assert len(resolved) == 2
        assert resolved[0].text == "Definition 1"
        assert resolved[1].text == "Definition 3"

    def test_acronym_resolver_resolves_with_default_k(self, acronym_resolver, acronym):
        """
        Test AcronymResolver uses the default top_k value (5).
        """
        resolved = acronym_resolver.resolve(acronym)
        assert len(resolved) == 3  # Since we have 3 definitions in the mock lookup

    def test_acronym_instantiation(self):
        """
        Test that Acronym object is correctly instantiated.
        """
        acronym = Acronym(text="HTML")
        assert acronym.text == "HTML"
