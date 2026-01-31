import pytest

from plainera_unacronym.nlp.extraction.matchers.defs.common import build_kept_phrase


class TestBuildKeptPhrase:
    def test_keeps_hit_tokens_only_when_no_bridges_or_numeric(self, _patch):
        # Make numeric-leading always False to keep this test tight.
        _patch(build_kept_phrase, _numeric_leading=lambda tok, include_numeric_leading: False)

        tokens = ["Portable", "Document", "Format"]
        out = build_kept_phrase(
            tokens,
            tok_left=0,
            tok_right=2,
            hit_tokens={0, 2},
            bridges=set(),
            include_numeric_leading=True,
        )
        assert out == "Portable Format"

    def test_keeps_bridge_tokens_case_insensitive(self, _patch):
        _patch(build_kept_phrase, _numeric_leading=lambda tok, include_numeric_leading: False)

        tokens = ["Ministry", "of", "Magic"]
        out = build_kept_phrase(
            tokens,
            tok_left=0,
            tok_right=2,
            hit_tokens={0, 2},
            bridges={"of"},
            include_numeric_leading=True,
        )
        assert out == "Ministry of Magic"

    def test_keeps_numeric_leading_tokens_when_enabled(self, _patch):
        # Treat tokens that start with a digit as numeric-leading.
        _patch(build_kept_phrase, _numeric_leading=lambda tok, include_numeric_leading: include_numeric_leading and tok[:1].isdigit())

        tokens = ["3M", "Portable", "format"]
        out = build_kept_phrase(
            tokens,
            tok_left=0,
            tok_right=2,
            hit_tokens={1, 2},
            bridges=set(),
            include_numeric_leading=True,
        )
        assert out == "3M Portable format"

    def test_does_not_keep_numeric_leading_tokens_when_disabled(self, _patch):
        _patch(build_kept_phrase, _numeric_leading=lambda tok, include_numeric_leading: include_numeric_leading and tok[:1].isdigit())

        tokens = ["3M", "Portable", "format"]
        out = build_kept_phrase(
            tokens,
            tok_left=0,
            tok_right=2,
            hit_tokens={1, 2},
            bridges=set(),
            include_numeric_leading=False,
        )
        assert out == "Portable format"

    def test_falls_back_to_full_window_when_nothing_qualifies(self, _patch):
        _patch(build_kept_phrase, _numeric_leading=lambda tok, include_numeric_leading: False)

        tokens = ["Portable", "Document", "Format"]
        out = build_kept_phrase(
            tokens,
            tok_left=0,
            tok_right=2,
            hit_tokens=set(),
            bridges=set(),
            include_numeric_leading=True,
        )
        assert out == "Portable Document Format"

    def test_collapses_whitespace_and_strips_trailing_punct(self, _patch):
        _patch(build_kept_phrase, _numeric_leading=lambda tok, include_numeric_leading: False)

        tokens = ["Graphics", "  Processing", "Unit..."]
        out = build_kept_phrase(
            tokens,
            tok_left=0,
            tok_right=2,
            hit_tokens={0, 1, 2},
            bridges=set(),
            include_numeric_leading=True,
        )
        assert out == "Graphics Processing Unit"

    def test_raises_value_error_when_tok_left_greater_than_tok_right(self):
        tokens = ["a", "b", "c"]
        with pytest.raises(ValueError):
            build_kept_phrase(
                tokens,
                tok_left=2,
                tok_right=1,
                hit_tokens={1},
                bridges=set(),
            )

    def test_raises_index_error_when_window_out_of_range(self, _patch):
        _patch(build_kept_phrase, _numeric_leading=lambda tok, include_numeric_leading: False)

        tokens = ["a", "b"]
        with pytest.raises(IndexError):
            build_kept_phrase(
                tokens,
                tok_left=0,
                tok_right=2,
                hit_tokens={0},
                bridges=set(),
            )


class TestBuildKeptPhraseIntegration:
    def test_keeps_hit_tokens_and_bridge_tokens(self):
        tokens = ["Ministry", "of", "Magic"]
        out = build_kept_phrase(
            tokens,
            tok_left=0,
            tok_right=2,
            hit_tokens={0, 2},
            bridges={"of"},
            include_numeric_leading=True,
        )
        assert out == "Ministry of Magic"

    def test_keeps_numeric_leading_token_when_enabled(self):
        # "3M" is numeric-leading (first alnum is '3' -> not alpha), so it should be kept.
        tokens = ["3M", "Portable", "format"]
        out = build_kept_phrase(
            tokens,
            tok_left=0,
            tok_right=2,
            hit_tokens={1, 2},
            bridges=set(),
            include_numeric_leading=True,
        )
        assert out == "3M Portable format"

    def test_drops_numeric_leading_token_when_disabled(self):
        tokens = ["3M", "Portable", "format"]
        out = build_kept_phrase(
            tokens,
            tok_left=0,
            tok_right=2,
            hit_tokens={1, 2},
            bridges=set(),
            include_numeric_leading=False,
        )
        assert out == "Portable format"

    def test_falls_back_to_full_window_when_nothing_qualifies(self):
        tokens = ["Portable", "Document", "Format"]
        out = build_kept_phrase(
            tokens,
            tok_left=0,
            tok_right=2,
            hit_tokens=set(),
            bridges=set(),
            include_numeric_leading=True,
        )
        assert out == "Portable Document Format"

    def test_collapses_whitespace_and_strips_trailing_punct(self):
        # collapse_ws should squash multi-space, strip_trailing_punct_str should remove trailing punctuation.
        tokens = ["Graphics", "Processing", "Unit..."]
        out = build_kept_phrase(
            tokens,
            tok_left=0,
            tok_right=2,
            hit_tokens={0, 1, 2},
            bridges=set(),
            include_numeric_leading=True,
        )
        assert out == "Graphics Processing Unit"

    def test_window_subset_keeps_only_qualifiers_inside_window(self):
        tokens = ["Alpha", "of", "Beta", "Gamma"]
        out = build_kept_phrase(
            tokens,
            tok_left=1,
            tok_right=3,
            hit_tokens={2},
            bridges={"of"},
            include_numeric_leading=True,
        )
        # Only indices 1..3 are considered; within that window keep "of" (bridge) and "Beta" (hit)
        assert out == "of Beta"
