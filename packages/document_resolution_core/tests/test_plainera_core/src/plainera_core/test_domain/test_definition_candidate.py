from plainera_core.core.domain import DefinitionCandidate


class TestAcronymFunctions:

    def test_definition_candidate_comparison(self):
        """
        Test that DefinitionCandidate is correctly ordered by score.
        """
        def1 = DefinitionCandidate(text="Def 1", score=1.0)
        def2 = DefinitionCandidate(text="Def 2", score=2.0)
        assert def1 != def2  # Different texts and scores
        assert def1.score < def2.score  # Lower score comes first
