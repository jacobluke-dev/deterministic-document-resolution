import pytest
from plainera_unacronym.nlp.extraction.matchers.defs import find_parenthetical_longform_before_acr


class TestFindParentheticalLongformBeforeAcrIntegration:
    def test_basic_forward_phrase_then_acr(self, dummy_cfg):
        cfg = dummy_cfg()
        snippet = "Portable Document Format (PDF)"
        out = find_parenthetical_longform_before_acr(snippet, "PDF", cfg)
        assert len(out) == 1
        assert out[0].definition == "Portable Document Format"

    def test_trailing_punctuation_is_normalized(self, dummy_cfg):
        cfg = dummy_cfg()
        snippet = "Graphics   Processing   Unit...   (GPU)"
        out = find_parenthetical_longform_before_acr(snippet, "GPU", cfg)
        assert len(out) == 1
        assert out[0].definition == "Graphics Processing Unit"

    def test_titlecase_tail_preference_is_respected(self, dummy_cfg):
        cfg = dummy_cfg()
        # If the tighten_definition_span favors the last TitleCase/UPPER chunk,
        # ensure we still get the meaningful tail.
        snippet = "See also the HyperText Transfer Protocol (HTTP)"
        out = find_parenthetical_longform_before_acr(snippet, "HTTP", cfg)
        assert len(out) == 1
        assert out[0].definition == "HyperText Transfer Protocol"

    def test_boundary_and_whitespace_variants(self, dummy_cfg):
        cfg = dummy_cfg()
        snippet = "Read Only Memory   ( ROM )   "
        out = find_parenthetical_longform_before_acr(snippet, "ROM", cfg)
        assert len(out) == 1
        assert out[0].definition == "Read Only Memory"

    def test_respects_max_chars_integration(self, dummy_cfg):
        cfg = dummy_cfg(max_phrase_chars=10)
        snippet = "Hypertext Transfer Protocol (HTTP)"
        # Def > 10 chars → no match
        assert find_parenthetical_longform_before_acr(snippet, "HTTP", cfg) == []
