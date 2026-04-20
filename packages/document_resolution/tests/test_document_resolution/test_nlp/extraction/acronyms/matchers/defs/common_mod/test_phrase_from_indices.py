import pytest
from document_resolution.nlp.extraction.acronyms.matchers.defs.defs_common import phrase_from_indices


class TestPhraseFromIndices:
    def test_joins_tokens_in_given_index_order(self):
        tokens = ["Portable", "Document", "Format"]
        assert phrase_from_indices(tokens, [0, 2]) == "Portable Format"

    def test_collapses_internal_whitespace(self):
        tokens = ["Graphics", "  Processing", "Unit"]
        # join introduces single spaces, but tokens may contain internal whitespace too
        assert phrase_from_indices(tokens, [0, 1, 2]) == "Graphics Processing Unit"

    def test_strips_trailing_punctuation_and_trailing_ws(self):
        tokens = ["Hello", "world!!!", "PDF,"]  # trailing punct should be removed only at end of phrase
        assert phrase_from_indices(tokens, [0, 1, 2]) == "Hello world!!! PDF"

    def test_trailing_punct_strip_only_applies_to_end(self):
        tokens = ["Wait,", "what?", "Really,"]
        # comma in middle remains; trailing comma removed
        assert phrase_from_indices(tokens, [0, 1, 2]) == "Wait, what? Really"

    def test_empty_indices_returns_empty_string(self):
        tokens = ["Portable", "Document", "Format"]
        assert phrase_from_indices(tokens, []) == ""

    def test_raises_index_error_when_index_out_of_range(self):
        tokens = ["Portable", "Document"]
        with pytest.raises(IndexError):
            phrase_from_indices(tokens, [0, 2])
