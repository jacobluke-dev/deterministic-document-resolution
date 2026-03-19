from plainera_unacronym.nlp.extraction.acronyms.matchers.defs.common import strip_inline_cue_prefix


class DummyCfg:
    def __init__(self, inline_cues):
        self.inline_cues = inline_cues


class TestStripInlineCuePrefix:
    def test_returns_none_when_no_inline_cues_configured(self):
        cfg = DummyCfg(inline_cues=())
        assert strip_inline_cue_prefix("stands for Portable Document Format", cfg) is None

    def test_returns_none_when_text_does_not_start_with_cue(self):
        cfg = DummyCfg(inline_cues=(r"stands for", r"means"))
        assert strip_inline_cue_prefix("Portable Document Format", cfg) is None

    def test_matches_simple_cue_and_returns_remaining_and_offset(self):
        cfg = DummyCfg(inline_cues=(r"stands for",))
        t = "stands for Portable Document Format"
        out = strip_inline_cue_prefix(t, cfg)
        assert out is not None
        rem, off = out
        assert rem == "Portable Document Format"
        assert t[off:] == rem

    def test_is_case_insensitive(self):
        cfg = DummyCfg(inline_cues=(r"stands for",))
        t = "StAnDs FoR Portable Document Format"
        rem, off = strip_inline_cue_prefix(t, cfg)
        assert rem == "Portable Document Format"
        assert t[off:] == rem

    def test_allows_leading_whitespace_and_optional_comma(self):
        cfg = DummyCfg(inline_cues=(r"stands for",))
        t = "   ,   stands for   Portable Document Format"
        rem, off = strip_inline_cue_prefix(t, cfg)

        assert rem == "Portable Document Format"
        assert t[off:] == rem

    def test_requires_space_after_cue(self):
        cfg = DummyCfg(inline_cues=(r"stands for",))
        # No whitespace after cue => should not match because regex uses \s+ after cue
        assert strip_inline_cue_prefix("stands forPortable", cfg) is None

    def test_tries_cues_in_order_first_match_wins(self):
        cfg = DummyCfg(inline_cues=(r"stands for", r"stands for sure"))
        t = "stands for sure Portable"
        # First cue "stands for" matches earliest and will win.
        rem, off = strip_inline_cue_prefix(t, cfg)
        assert rem == "sure Portable"
        assert t[off:] == rem

    def test_can_match_regex_cue_fragments(self):
        cfg = DummyCfg(inline_cues=(r"(?:stands\s+for|means)",))
        t = "means Portable Document Format"
        rem, off = strip_inline_cue_prefix(t, cfg)
        assert rem == "Portable Document Format"
        assert t[off:] == rem
