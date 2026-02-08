import pytest
from plainera_unacronym.nlp.extraction.matchers.defs.common import is_acronym_parenthetical_with_tail


class TestIsAcronymParentheticalWithTail:
    @pytest.mark.parametrize(
        "snippet, acr",
        [
            ("(PDF, see Appendix A)", "PDF"),
            ('("PDF": see Appendix A)', "PDF"),
            ("('PDF' - see Appendix A)", "PDF"),
            ("(PDF—see Appendix A)", "PDF"),
            ("(PDF – see Appendix A)", "PDF"),
            ("(PDF; see Appendix A)", "PDF"),
        ],
    )
    def test_true_for_acronym_parenthetical_with_tail(self, snippet, acr):
        assert is_acronym_parenthetical_with_tail(snippet, acr) is True

    @pytest.mark.parametrize(
        "snippet, acr",
        [
            ("(PDF)", "PDF"),
            ("(Portable Document Format)", "PDF"),
            ("(PDF )", "PDF"),
            ("(PDF, )", "PDF"),  # needs a non-whitespace tail token
        ],
    )
    def test_false_for_parenthetical_without_tail(self, snippet, acr):
        assert is_acronym_parenthetical_with_tail(snippet, acr) is False
