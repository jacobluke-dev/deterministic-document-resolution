import pytest
from document_resolution.nlp.extraction.acronyms.anchored.normalise import tighten_definition_span


class TestTightenDefinitionSpan:
    def test_empty_returns_empty(self):
        assert tighten_definition_span("") == ""
        assert tighten_definition_span("   \n\t") == ""

    def test_keeps_titlecase_with_per(self):
        s = "Cost per Acquisition"
        assert tighten_definition_span(s) == "Cost per Acquisition"

    def test_keeps_titlecase_with_common_linkers(self):
        s = "Department of Health and Social Care"
        assert tighten_definition_span(s) == "Department of Health and Social Care"

    def test_keeps_ampersand(self):
        s = "Research & Development"
        assert tighten_definition_span(s) == "Research & Development"

    def test_all_caps_phrase_is_preserved(self):
        s = "COST PER ACQUISITION"
        assert tighten_definition_span(s) == "COST PER ACQUISITION"

    def test_all_caps_token_is_allowed(self):
        s = "something before. NHS Guidance"
        assert tighten_definition_span(s) == "NHS Guidance"

    def test_prefers_rightmost_titlecase_run_when_tail_is_title_run(self):
        s = "Noise before. Portable Document Format. Read Only Memory."
        assert tighten_definition_span(s) == "Read Only Memory"

    def test_prefers_last_titlecase_run_at_end(self):
        s = "Some intro text, Department of Education and Skills"
        assert tighten_definition_span(s) == "Department of Education and Skills"

    def test_lower_hyphen_tail_does_not_win_over_full_title_run(self):
        s = "Single sign-on"
        assert tighten_definition_span(s) == "Single sign-on"

    def test_lower_hyphen_inside_title_run_is_kept(self):
        s = "Single sign-on for authentication"
        assert tighten_definition_span(s) == "Single sign-on"

    def test_falls_back_to_last_clause_then_finds_run_in_tail(self):
        s = "nonsense: ignore this; Portable Document Format"
        assert tighten_definition_span(s) == "Portable Document Format"

    def test_head_run_in_tail_is_kept_only(self):
        s = "blah blah. Portable Document Format is widely used"
        assert tighten_definition_span(s) == "Portable Document Format"

    def test_fallback_when_no_titlecase_run(self):
        s = "this is a lowercase tail with numbers 123"
        assert tighten_definition_span(s) == "this is a lowercase tail with numbers 123"

    def test_final_fallback_returns_cleaned_tail_when_no_title_runs(self):
        s = "this is not titlecase; just words, maybe."
        assert tighten_definition_span(s) == "maybe"

    def test_handles_unicode_apostrophes_and_dashes(self):
        s = "Director-General’s Office – North"
        s2 = f"See memo for {s}"
        assert tighten_definition_span(s2) == "Director-General’s Office – North"

    def test_works_when_titlecase_is_after_a_boundary(self):
        s = "some preface. Cost per Acquisition"
        assert tighten_definition_span(s) == "Cost per Acquisition"

    def test_picks_titlecase_run_for_pto_sentence(self):
        s = "Please Turn Over on print jobs."
        assert tighten_definition_span(s) == "Please Turn Over"

    @pytest.mark.parametrize(
        "s,expected",
        [
            ("Cost per Acquisition,", "Cost per Acquisition"),
            ("Portable Document Format.", "Portable Document Format"),
            ("Portable Document Format   ...   ", "Portable Document Format"),
            ("Portable Document Format)", "Portable Document Format"),
            ("Portable Document Format»  ", "Portable Document Format"),
        ],
    )
    def test_trims_trailing_punct_and_collapses_ws(self, s, expected):
        assert tighten_definition_span(s) == expected
