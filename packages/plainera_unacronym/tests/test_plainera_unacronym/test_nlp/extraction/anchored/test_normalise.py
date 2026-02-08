import pytest
from plainera_unacronym.nlp.extraction.anchored.normalise import tighten_definition_span


class TestTightenDefinitionSpan:
    def test_empty_returns_empty(self):
        assert tighten_definition_span("") == ""
        assert tighten_definition_span("   \n\t") == ""

    def test_prefers_rightmost_titlecase_run_when_tail_is_title_run(self):
        # Make the last clause *be* the run (no leading junk like "then")
        s = "Noise before. Portable Document Format. Read Only Memory."
        assert tighten_definition_span(s) == "Read Only Memory"

    def test_lower_hyphen_tail_does_not_win_over_full_title_run(self):
        # This is the regression your NEW guard actually targets:
        # rightmost match would otherwise be "sign-on", but should return "Single sign-on".
        s = "Single sign-on"
        assert tighten_definition_span(s) == "Single sign-on"

    def test_lower_hyphen_inside_title_run_is_kept(self):
        s = "Single sign-on for authentication"
        # Tail starts with title run "Single sign-on" then stops at "for"
        assert tighten_definition_span(s) == "Single sign-on"

    def test_falls_back_to_last_clause_then_finds_run_in_tail(self):
        # No run across whole string should be used; last clause contains the run.
        s = "nonsense: ignore this; Portable Document Format"
        assert tighten_definition_span(s) == "Portable Document Format"

    def test_head_run_in_tail_is_kept_only(self):
        # 2a: if tail starts with a TitleCase run, keep only that run.
        s = "blah blah. Portable Document Format is widely used"
        # Tail begins with "Portable Document Format" then continues; keep only the run.
        assert tighten_definition_span(s) == "Portable Document Format"

    def test_final_fallback_returns_cleaned_tail_when_no_title_runs(self):
        s = "this is not titlecase; just words, maybe."
        assert tighten_definition_span(s) == "maybe"

    @pytest.mark.parametrize(
        "s,expected",
        [
            ("Portable Document Format.", "Portable Document Format"),
            ("Portable Document Format   ...   ", "Portable Document Format"),
            ("Portable Document Format)", "Portable Document Format"),
            ("Portable Document Format»  ", "Portable Document Format"),
        ],
    )
    def test_trims_trailing_punct_and_collapses_ws(self, s, expected):
        assert tighten_definition_span(s) == expected

    def test_all_caps_token_is_allowed(self):
        s = "something before. NHS Guidance"
        assert tighten_definition_span(s) == "NHS Guidance"
