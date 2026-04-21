from document_resolution.nlp.extraction.acronyms.matchers.defs.defs_common import expand_numeric_leading_window


class TestExpandNumericLeadingWindow:
    def test_expands_left_and_right_over_adjacent_numeric_leading_tokens(self, _patch):
        def fake_first_alnum_char_upper(tok: str):
            for ch in tok:
                if ch.isalnum():
                    return ch.upper()
            return None

        _patch(expand_numeric_leading_window, first_alnum_char_upper=fake_first_alnum_char_upper)

        tokens = ["3M", "Portable", "Format", "2", "PDF"]
        # window covering "Portable Format" should grow to include "3M" on the left and "2" on the right
        assert expand_numeric_leading_window(tokens, tok_left=1, tok_right=2) == (0, 3)

    def test_does_not_expand_when_adjacent_tokens_are_not_numeric_leading(self, _patch):
        def fake_first_alnum_char_upper(tok: str):
            for ch in tok:
                if ch.isalnum():
                    return ch.upper()
            return None

        _patch(expand_numeric_leading_window, first_alnum_char_upper=fake_first_alnum_char_upper)

        tokens = ["Alpha", "Beta", "Gamma"]
        assert expand_numeric_leading_window(tokens, tok_left=1, tok_right=1) == (1, 1)


class TestExpandNumericLeadingWindowIntegration:
    def test_expands_left_and_right_over_adjacent_numeric_leading_tokens(self):
        tokens = ["3M", "Portable", "Format", "2", "PDF"]

        # window covering "Portable Format" should grow to include "3M" on the left and "2" on the right
        assert expand_numeric_leading_window(tokens, tok_left=1, tok_right=2) == (0, 3)

    def test_does_not_expand_when_adjacent_tokens_are_not_numeric_leading(self):
        tokens = ["Alpha", "Beta", "Gamma"]

        assert expand_numeric_leading_window(tokens, tok_left=1, tok_right=1) == (1, 1)

    def test_expands_over_punct_wrapped_numeric_leading_tokens(self):
        # This checks real-world tokenisation artefacts (parens / punctuation stuck to token).
        # If the first_alnum_char_upper strips punctuation, these should still count numeric-leading.
        tokens = ["(3M)", "Portable", "Format", "2)", "PDF"]

        assert expand_numeric_leading_window(tokens, tok_left=1, tok_right=2) == (0, 3)

    def test_expands_multiple_steps_until_non_numeric_leading(self):
        tokens = ["1", "2", "Portable", "Format", "3", "4", "PDF"]

        # Start at "Portable Format" => should pull in 1,2 on the left and 3,4 on the right
        assert expand_numeric_leading_window(tokens, tok_left=2, tok_right=3) == (0, 5)

    def test_no_expansion_at_edges(self):
        tokens = ["3M", "Portable", "Format"]

        # Already at left edge; should not underflow
        assert expand_numeric_leading_window(tokens, tok_left=0, tok_right=1) == (0, 1)

        tokens2 = ["Portable", "Format", "2"]
        # Already at right edge; should not overflow
        assert expand_numeric_leading_window(tokens2, tok_left=0, tok_right=1) == (0, 2)


class TestNumericLeadingUsedByExpandNumericLeadingWindow:
    def test_expand_window_pulls_in_numeric_neighbors(self):
        tokens = ["3M", "Portable", "Format", "2", "PDF"]
        assert expand_numeric_leading_window(tokens, tok_left=1, tok_right=2) == (0, 3)
