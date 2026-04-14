import pytest
from document_resolution.nlp.extraction.acronyms.matchers.numeric_matcher import (
    WORD_TO_DIGITS,
    consume_left_numeric_designator,
)


class TestConsumeLeftNumericDesignator:
    # ----------------------------
    # 1) Guard rails / early exits
    # ----------------------------

    @pytest.mark.parametrize(
        "acr,tokens,tok_left,expected",
        [
            ("", ["fifth", "generation"], 1, 1),  # not acr
            ("GPU", ["fifth", "generation"], 1, 1),  # acr has no leading digits
            ("5G", ["fifth", "generation"], 0, 0),  # tok_left <= 0
            ("5G", [], 0, 0),  # empty tokens; tok_left <= 0 hits first guard
        ],
    )
    def test_guard_clauses(self, acr, tokens, tok_left, expected):
        got = consume_left_numeric_designator(
            acr=acr,
            tokens=tokens,
            tok_left=tok_left,
            word_to_digits=WORD_TO_DIGITS,
        )
        assert got == expected

    # -----------------------------------------
    # 2) Normalisation: edge punct + hyphen head
    # -----------------------------------------

    @pytest.mark.parametrize(
        "acr,tokens,tok_left,expected",
        [
            ("5G", ["(5th)", "generation"], 1, 0),  # edge punct stripped -> "5th"
            ("5G", ["'5th'", "generation"], 1, 0),  # edge punct stripped -> "5th"
            ("5G", ["“5th”", "generation"], 1, 0),  # edge punct stripped -> "5th"
            ("5G", ["5th-generation", "(5G)"], 1, 0),  # hyphen head -> "5th"
        ],
    )
    def test_normalisation_prev_token(self, acr, tokens, tok_left, expected):
        got = consume_left_numeric_designator(
            acr=acr,
            tokens=tokens,
            tok_left=tok_left,
            word_to_digits=WORD_TO_DIGITS,
        )
        assert got == expected

    # -------------------------------------------
    # 3) Matching modes: numeric ordinal / numeric
    # -------------------------------------------

    @pytest.mark.parametrize(
        "acr,tokens,tok_left,expected",
        [
            # numeric ordinal (case-insensitive)
            ("5G", ["5th", "generation"], 1, 0),
            ("12V", ["12th", "edition"], 1, 0),
            ("12V", ["12TH", "edition"], 1, 0),
            # plain numeric
            ("5G", ["5", "generation"], 1, 0),
            ("12V", ["12", "volt"], 1, 0),
            # multi-digit leading run (acr "12V" wants "12")
            ("12V", ["twelfth", "edition"], 1, 0),
        ],
    )
    def test_matches_numeric_forms(self, acr, tokens, tok_left, expected):
        got = consume_left_numeric_designator(
            acr=acr,
            tokens=tokens,
            tok_left=tok_left,
            word_to_digits=WORD_TO_DIGITS,
        )
        assert got == expected

    # --------------------------------------
    # 4) Matching modes: word -> digit mapping
    # --------------------------------------

    @pytest.mark.parametrize(
        "acr,tokens,tok_left,expected",
        [
            ("5G", ["fifth", "generation"], 1, 0),
            ("10G", ["tenth", "generation"], 1, 0),
            ("5G", ["five", "generation"], 1, 0),  # cardinal mapping enabled in WORD_TO_DIGITS
        ],
    )
    def test_matches_word_mappings(self, acr, tokens, tok_left, expected):
        got = consume_left_numeric_designator(
            acr=acr,
            tokens=tokens,
            tok_left=tok_left,
            word_to_digits=WORD_TO_DIGITS,
        )
        assert got == expected

    # ----------------------------------------------------
    # 5) Dot heuristic: don't consume if it's sentence-start
    # ----------------------------------------------------

    @pytest.mark.parametrize(
        "acr,tokens,tok_left",
        [
            ("5G", ["5th.", "Generation"], 1),  # '.' + capital => boundary => do not consume
        ],
    )
    def test_does_not_consume_when_trailing_dot_is_boundary(self, acr, tokens, tok_left):
        got = consume_left_numeric_designator(
            acr=acr,
            tokens=tokens,
            tok_left=tok_left,
            word_to_digits=WORD_TO_DIGITS,
        )
        assert got == tok_left

    @pytest.mark.parametrize(
        "acr,tokens,tok_left,expected",
        [
            ("5G", ["5th.", "generation"], 1, 0),  # '.' + lowercase => not boundary => consume
        ],
    )
    def test_consumes_when_trailing_dot_not_boundary(self, acr, tokens, tok_left, expected):
        got = consume_left_numeric_designator(
            acr=acr,
            tokens=tokens,
            tok_left=tok_left,
            word_to_digits=WORD_TO_DIGITS,
        )
        assert got == expected

    # ----------------------------
    # 6) Negative matches (mismatch)
    # ----------------------------

    @pytest.mark.parametrize(
        "acr,tokens,tok_left",
        [
            ("5G", ["sixth", "generation"], 1),
            ("12V", ["11th", "edition"], 1),
            ("5G", ["4", "generation"], 1),
            ("5G", ["B5", "generation"], 1),
        ],
    )
    def test_does_not_consume_when_prev_does_not_match_number(self, acr, tokens, tok_left):
        got = consume_left_numeric_designator(
            acr=acr,
            tokens=tokens,
            tok_left=tok_left,
            word_to_digits=WORD_TO_DIGITS,
        )
        assert got == tok_left
