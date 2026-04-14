import pytest
from plainera_core.core.domain import Acronym, DefinitionCandidate
from plainera_core.core.services.resolver import AcronymResolver


@pytest.fixture
def empty_lookup():
    def _lookup(query):
        return []
    return _lookup


@pytest.fixture
def empty_acronym_resolver(empty_lookup):
    return AcronymResolver(empty_lookup)


@pytest.fixture
def tied_lookup():
    def _lookup(query):
        return [
            DefinitionCandidate(text="Definition A", score=10.0),
            DefinitionCandidate(text="Definition B", score=10.0),
            DefinitionCandidate(text="Definition C", score=8.0),
        ]
    return _lookup


@pytest.fixture
def tied_acronym_resolver(tied_lookup):
    return AcronymResolver(tied_lookup)

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
        assert [r.text for r in resolved] == ["Definition 1", "Definition 3", "Definition 2"]

    def test_acronym_instantiation(self):
        """
        Test that Acronym object is correctly instantiated.
        """
        acronym = Acronym(text="HTML")
        assert acronym.text == "HTML"

    def test_acronym_resolver_returns_empty_list_when_no_matches(self, empty_acronym_resolver, acronym):
        resolved = empty_acronym_resolver.resolve(acronym)
        assert resolved == []

    def test_acronym_resolver_respects_top_k_one(self, acronym_resolver, acronym):
        resolved = acronym_resolver.resolve(acronym, top_k=1)
        assert len(resolved) == 1
        assert resolved[0].text == "Definition 1"

    def test_acronym_resolver_caps_at_available_results(self, acronym_resolver, acronym):
        resolved = acronym_resolver.resolve(acronym, top_k=10)
        assert len(resolved) == 3

    def test_acronym_resolver_orders_results_by_score_desc(self, acronym_resolver, acronym):
        resolved = acronym_resolver.resolve(acronym, top_k=3)
        scores = [r.score for r in resolved]
        assert scores == sorted(scores, reverse=True)

    def test_acronym_resolver_is_deterministic_on_tied_scores(self, tied_acronym_resolver, acronym):
        resolved_1 = tied_acronym_resolver.resolve(acronym, top_k=3)
        resolved_2 = tied_acronym_resolver.resolve(acronym, top_k=3)

        assert [r.text for r in resolved_1] == [r.text for r in resolved_2]

    def test_acronym_resolver_handles_case_insensitive_lookup(self, acronym_resolver):
        resolved = acronym_resolver.resolve(Acronym(text="api"))
        assert len(resolved) > 0

    def test_acronym_resolver_returns_empty_list_for_non_positive_top_k(self, acronym_resolver, acronym):
        resolved = acronym_resolver.resolve(acronym, top_k=0)
        assert resolved == []


    def test_acronym_resolver_returns_top_k_ranked_results(self, acronym_resolver, acronym):
        resolved = acronym_resolver.resolve(acronym, top_k=2)
        assert len(resolved) == 2
        assert [r.text for r in resolved] == ["Definition 1", "Definition 3"]

    def test_acronym_resolver_uses_default_top_k(self, acronym_resolver, acronym):
        resolved = acronym_resolver.resolve(acronym)
        assert len(resolved) == 3
        assert [r.text for r in resolved] == ["Definition 1", "Definition 3", "Definition 2"]
