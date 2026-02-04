import pytest

from plainera_unacronym.nlp.extraction.core.normalise import tighten_label


class TestTightenLabel:
    def test_forward_stands_for(self):
        s = "PDF stands for Portable Document Format"
        assert tighten_label(s) == "Portable Document Format"

    def test_forward_means(self):
        s = "GPU means Graphics Processing Unit"
        assert tighten_label(s) == "Graphics Processing Unit"

    def test_forward_is(self):
        s = "ROM is Read Only Memory"
        assert tighten_label(s) == "Read Only Memory"

    def test_forward_are(self):
        s = "HTTP headers are Hypertext Transfer Protocol headers"
        assert tighten_label(s) == "Hypertext Transfer Protocol headers"

    def test_trailing_proper_noun_chunk_wins(self):
        # Proper chunk rule runs before splitter logic.
        s = "The non-profit North American Saxophone Alliance"
        assert tighten_label(s) == "North American Saxophone Alliance"

    def test_article_removed_when_no_proper_chunk(self):
        s = "The graphics processing unit"
        assert tighten_label(s) == "graphics processing unit"

    def test_leading_connectors_removed_twice_then_article_removed(self):
        # Connector stripping runs twice; then article stripping happens later.
        s = "And, which the Portable Document Format"
        assert tighten_label(s) == "Portable Document Format"

    def test_handles_hyphens_and_apostrophes_in_proper_chunk(self):
        # Regex allows letters, digits, apostrophes and hyphens inside words
        s = "The British-Irish Council"
        assert tighten_label(s) == "British-Irish Council"

        s2 = "Queen’s Award for Enterprise"
        # Proper-noun chunk is the trailing capitalised sequence:
        # "Queen’s Award for Enterprise" -> last proper chunk = "Enterprise"? No:
        # the trailing chunk matched should be the last Capitalised+ words sequence.
        # Use a more deterministic phrasing to ensure multi-word match:
        s2 = "The Queen’s Award"
        assert tighten_label(s2) == "Queen's Award"

    def test_no_change_when_already_minimal(self):
        s = "efficient data structure"
        assert tighten_label(s) == "efficient data structure"

    def test_mixed_case_non_proper_phrase_keeps_case_post_article_drop(self):
        s = "An adaptive threshold"
        # No trailing Proper-Noun chunk; article removed, rest kept
        assert tighten_label(s) == "adaptive threshold"

    def test_trailing_punct_trimmed_by_normaliser(self):
        s = "PDF stands for Portable Document Format..."
        assert tighten_label(s) == "Portable Document Format"

    def test_collapses_whitespace(self):
        s = "PDF   stands   for   Portable   Document   Format"
        assert tighten_label(s) == "Portable Document Format"


class TestTightenLabelConnectorStability:
    @pytest.mark.parametrize(
        "s, expected",
        [
            ("And the Portable Document Format", "Portable Document Format"),
            ("Which: the Portable Document Format", "Portable Document Format"),
            ("For, the Portable Document Format", "Portable Document Format"),
        ],
    )
    def test_common_connectors_strip(self, s, expected):
        assert tighten_label(s) == expected
