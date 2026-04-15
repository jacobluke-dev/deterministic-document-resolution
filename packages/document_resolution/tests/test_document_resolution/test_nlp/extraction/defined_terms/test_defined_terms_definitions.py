from __future__ import annotations

from document_resolution.nlp.extraction.defined_terms.definitions import (
    _extract_parenthetical_alias_target,
    _find_definition_end,
)


class TestFindDefinitionEnd:
    def test_stops_at_period(self):
        text = "the date on which both Parties sign this Agreement. Next sentence."
        start = 0

        end = _find_definition_end(text, start)

        assert end == text.index(".")
        assert text[start:end] == "the date on which both Parties sign this Agreement"

    def test_stops_at_semicolon(self):
        text = "the consultancy services described in the main body; additional text"
        start = 0

        end = _find_definition_end(text, start)

        assert end == text.index(";")
        assert text[start:end] == "the consultancy services described in the main body"

    def test_stops_at_blank_line(self):
        text = "the software maintenance services described in this Schedule\n\nSchedule A"
        start = 0

        end = _find_definition_end(text, start)

        assert end == text.index("\n\n")
        assert text[start:end] == "the software maintenance services described in this Schedule"

    def test_returns_chunk_end_when_no_boundary_found_within_max_chars(self):
        text = "abcdefghijABCDEFGHIJ"
        start = 0

        end = _find_definition_end(text, start, max_chars=10)

        assert end == 10
        assert text[start:end] == "abcdefghij"

    def test_returns_none_for_whitespace_only_chunk(self):
        text = "     "
        start = 0

        end = _find_definition_end(text, start)

        assert end is None

    def test_returns_none_when_start_is_at_or_beyond_text_length(self):
        text = "abc"

        assert _find_definition_end(text, 3) is None
        assert _find_definition_end(text, 10) is None

class TestExtractParentheticalAliasTarget:
    def test_extracts_antecedent_phrase_for_agreement_alias(self):
        text = 'This Master Services Agreement (the "Agreement") is entered into on the Effective Date.'
        intro_span = ("Agreement", 37, 46)

        span, alias_target_text = _extract_parenthetical_alias_target(text, intro_span)

        assert alias_target_text == "This Master Services Agreement"
        assert span is not None
        assert span == ("This Master Services Agreement", 0, 30)
        assert text[span[1]:span[2]] == "This Master Services Agreement"

    def test_extracts_party_name_for_supplier_alias(self):
        text = 'Acme Limited (the "Supplier") shall provide the Services.'
        intro_span = ("Supplier", 19, 27)

        span, alias_target_text = _extract_parenthetical_alias_target(text, intro_span)

        assert alias_target_text == "Acme Limited"
        assert span is not None
        assert span == ("Acme Limited", 0, 12)
        assert text[span[1]:span[2]] == "Acme Limited"

    def test_extracts_schedule_style_antecedent(self):
        text = 'Schedule A (the "Service Levels Schedule") forms part of this Agreement.'
        intro_span = ("Service Levels Schedule", 17, 40)

        span, alias_target_text = _extract_parenthetical_alias_target(text, intro_span)

        assert alias_target_text == "Schedule A"
        assert span is not None
        assert span == ("Schedule A", 0, 10)
        assert text[span[1]:span[2]] == "Schedule A"

    def test_stops_at_newline_boundary(self):
        text = 'Background text.\nAcme Limited (the "Supplier") shall provide the Services.'
        intro_span = ("Supplier", 36, 44)

        span, alias_target_text = _extract_parenthetical_alias_target(text, intro_span)

        assert alias_target_text == "Acme Limited"
        assert span is not None
        assert text[span[1]:span[2]] == "Acme Limited"

    def test_returns_none_when_opening_parenthesis_not_found(self):
        text = 'This Master Services Agreement the "Agreement" is entered into on the Effective Date.'
        intro_span = ("Agreement", 36, 45)

        span, alias_target_text = _extract_parenthetical_alias_target(text, intro_span)

        assert span is None
        assert alias_target_text is None

    def test_returns_none_when_nothing_precedes_parenthesis(self):
        text = '(the "Agreement") is entered into on the Effective Date.'
        intro_span = ("Agreement", 6, 15)

        span, alias_target_text = _extract_parenthetical_alias_target(text, intro_span)

        assert span is None
        assert alias_target_text is None

    def test_trims_whitespace_before_parenthetical_alias(self):
        text = 'Acme Limited   (the "Supplier") shall provide the Services.'
        intro_span = ("Supplier", 21, 29)

        span, alias_target_text = _extract_parenthetical_alias_target(text, intro_span)

        assert alias_target_text == "Acme Limited"
        assert span is not None
        assert span == ("Acme Limited", 0, 12)
        assert text[span[1]:span[2]] == "Acme Limited"

    def test_does_not_include_trailing_prose_after_alias(self):
        text = 'This Master Services Agreement (the "Agreement") is entered into on the Effective Date.'
        intro_span = ("Agreement", 37, 46)

        span, alias_target_text = _extract_parenthetical_alias_target(text, intro_span)

        assert alias_target_text == "This Master Services Agreement"
        assert span is not None
        assert "is entered into on the Effective Date" not in alias_target_text
