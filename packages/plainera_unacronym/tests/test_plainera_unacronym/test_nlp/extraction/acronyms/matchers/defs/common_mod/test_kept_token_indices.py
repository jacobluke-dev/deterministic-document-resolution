from plainera_unacronym.nlp.extraction.acronyms.matchers.defs.common import kept_token_indices


class TestKeptTokenIndices:
    def test_keeps_hit_tokens_inside_window(self):
        tokens = ["Portable", "Document", "Format"]
        out = kept_token_indices(
            tokens,
            tok_left=0,
            tok_right=2,
            hit_tokens={0, 2},
            bridges=set(),
            include_numeric_leading=False,
        )
        assert out == [0, 2]

    def test_keeps_bridge_tokens_inside_window(self):
        tokens = ["Ministry", "of", "Magic"]
        out = kept_token_indices(
            tokens,
            tok_left=0,
            tok_right=2,
            hit_tokens={0, 2},
            bridges={"of"},
            include_numeric_leading=False,
        )
        assert out == [0, 1, 2]

    def test_keeps_numeric_leading_tokens_when_enabled(self):
        tokens = ["3M", "Portable", "Format", "2", "PDF"]
        out = kept_token_indices(
            tokens,
            tok_left=0,
            tok_right=3,  # window doesn't include "PDF"
            hit_tokens={1, 2},
            bridges=set(),
            include_numeric_leading=True,
        )
        # 3M and 2 are numeric-leading; 1,2 are hits
        assert out == [0, 1, 2, 3]

    def test_does_not_keep_numeric_leading_tokens_when_disabled(self):
        tokens = ["3M", "Portable", "Format", "2"]
        out = kept_token_indices(
            tokens,
            tok_left=0,
            tok_right=3,
            hit_tokens={1, 2},
            bridges=set(),
            include_numeric_leading=False,
        )
        assert out == [1, 2]

    def test_fallback_returns_full_window_when_no_hits_bridges_or_numeric(self):
        tokens = ["Portable", "Document", "Format"]
        out = kept_token_indices(
            tokens,
            tok_left=0,
            tok_right=2,
            hit_tokens=set(),
            bridges=set(),
            include_numeric_leading=False,
        )
        assert out == [0, 1, 2]

    def test_only_scans_within_tok_left_tok_right(self):
        tokens = ["X", "3M", "Portable", "Format", "2", "Y"]
        out = kept_token_indices(
            tokens,
            tok_left=2,
            tok_right=3,
            hit_tokens={2},
            bridges=set(),
            include_numeric_leading=True,
        )
        # numeric-leading tokens are outside window, so should not appear
        assert out == [2]


class TestKeptTokenIndicesIntegration:
    def test_realistic_bridge_and_numeric_kept_for_readability(self):
        tokens = ["3M", "Ministry", "of", "Magic"]
        kept = kept_token_indices(
            tokens,
            tok_left=0,
            tok_right=3,
            hit_tokens={1, 3},  # "Ministry", "Magic"
            bridges={"of"},
            include_numeric_leading=True,
        )
        assert kept == [0, 1, 2, 3]

    def test_realistic_window_without_bridges_keeps_only_hits_and_numeric(self):
        tokens = ["3M", "Portable", "Document", "Format", "2"]
        kept = kept_token_indices(
            tokens,
            tok_left=0,
            tok_right=4,
            hit_tokens={1, 2, 3},
            bridges={"of"},  # irrelevant here
            include_numeric_leading=True,
        )
        assert kept == [0, 1, 2, 3, 4]
