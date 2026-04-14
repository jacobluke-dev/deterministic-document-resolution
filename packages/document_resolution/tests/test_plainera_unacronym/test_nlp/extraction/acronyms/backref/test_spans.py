from plainera_unacronym.nlp.extraction.acronyms.backref.spans import best_span_by_initials, find_span_index, sent_spans


class TestBestSpanByInitials:
    def test_returns_none_for_empty_sentence(self):
        assert best_span_by_initials("PDF", "", max_chars=100) is None
        assert best_span_by_initials("PDF", "   \n\t  ", max_chars=100) is None

    def test_returns_none_when_acronym_has_no_letters(self):
        assert best_span_by_initials("123-._", "Portable Document Format", max_chars=100) is None

    def test_basic_match_returns_span(self):
        sent = "Portable Document Format (PDF) is common."
        assert best_span_by_initials("PDF", sent, max_chars=200) == "Portable Document Format"

    def test_case_insensitive_and_ignores_non_letters_in_acronym(self):
        sent = "Graphics Processing Unit is used."
        assert best_span_by_initials("g-p_u", sent, max_chars=200) == "Graphics Processing Unit"

    def test_prefers_shortest_token_window_when_multiple_candidates_exist(self):
        sent = "Portable Document Format appears, but later Portable Digital Format also appears."
        # Candidates:
        # - "Portable Document Format" (3 tokens)
        # - "Portable Digital Format" (3 tokens)
        # Same token length; tie-breaker is fewer chars => "Portable Digital Format"
        # (because 'Digital' shorter than 'Document')
        assert best_span_by_initials("PDF", sent, max_chars=200) == "Portable Digital Format"

    def test_tie_breaker_prefers_fewer_chars_for_same_token_count(self):
        sent = "Alpha Beta Gamma and Alpha B Gamma"
        # Both match "ABG" as 3-token spans; "Alpha B Gamma" is shorter in chars.
        assert best_span_by_initials("ABG", sent, max_chars=200) == "Alpha B Gamma"

    def test_max_chars_filters_out_candidates(self):
        sent = "Portable Document Format is common."
        # Candidate is longer than 10 chars -> rejected
        assert best_span_by_initials("PDF", sent, max_chars=10) is None
        # Large enough -> accepted
        assert best_span_by_initials("PDF", sent, max_chars=200) == "Portable Document Format"

    def test_non_alpha_leading_tokens_do_not_contribute_initials(self):
        sent = "3M Portable 7-Document Format reference."
        # Initials contributed are P and F (3M and 7-Document ignored)
        assert best_span_by_initials("PF", sent, max_chars=200) == "Portable 7-Document Format"
        # "PDF" cannot be formed because "7-Document" contributes no 'D'
        assert best_span_by_initials("PDF", sent, max_chars=200) is None

    def test_whitespace_is_collapsed_in_output(self):
        sent = "Portable   \n  Document\tFormat is common."
        assert best_span_by_initials("PDF", sent, max_chars=200) == "Portable Document Format"

    def test_returns_none_when_no_matching_span(self):
        sent = "Nothing relevant here."
        assert best_span_by_initials("PDF", sent, max_chars=200) is None


class TestSentSpansUnit:
    def test_empty_text(self):
        assert sent_spans("") == []

    def test_single_chunk_no_boundaries(self):
        text = "No sentence boundary here"
        assert sent_spans(text) == [(0, len(text))]

    def test_splits_on_period_followed_by_space(self):
        text = "One. Two."
        spans = sent_spans(text)
        # "One." ends at index 4, boundary is ". " (dot then space)
        assert [text[a:b] for (a, b) in spans] == ["One.", "Two."]

    def test_splits_on_question_and_exclamation(self):
        text = "What? Yes! Done."
        spans = sent_spans(text)
        assert [text[a:b] for (a, b) in spans] == ["What?", "Yes!", "Done."]

    def test_splits_on_ellipsis_char(self):
        text = "Wait… now."
        spans = sent_spans(text)
        assert [text[a:b] for (a, b) in spans] == ["Wait…", "now."]

    def test_splits_on_newlines_and_collapses_multiple_newlines(self):
        text = "First line\n\nSecond line\nThird"
        spans = sent_spans(text)
        assert [text[a:b] for (a, b) in spans] == ["First line", "Second line", "Third"]

    def test_does_not_emit_empty_spans_for_leading_or_trailing_separators(self):
        text = "\n\nHello.\n\n"
        spans = sent_spans(text)
        # Leading/trailing newlines should not create empty spans.
        assert [text[a:b] for (a, b) in spans] == ["Hello."]

    def test_spans_are_end_exclusive_and_cover_text(self):
        text = "A. B.\nC"
        spans = sent_spans(text)

        # End-exclusive slicing should reconstruct chunks
        chunks = [text[a:b] for (a, b) in spans]
        assert chunks == ["A.", "B.", "C"]

        # Spans should be strictly increasing and within bounds
        assert spans[0][0] == 0
        assert spans[-1][1] == len(text)
        for a, b in spans:
            assert 0 <= a < b <= len(text)

    def test_no_split_without_whitespace_after_punct(self):
        text = "One.Two"
        assert sent_spans(text) == [(0, len(text))]


class TestFindSpanIndexUnit:
    def test_returns_none_for_empty_spans(self):
        assert find_span_index([], 0) is None

    def test_finds_correct_span_for_position(self):
        spans = [(0, 3), (3, 7), (7, 10)]
        assert find_span_index(spans, 0) == 0
        assert find_span_index(spans, 2) == 0
        assert find_span_index(spans, 3) == 1
        assert find_span_index(spans, 6) == 1
        assert find_span_index(spans, 7) == 2
        assert find_span_index(spans, 9) == 2

    def test_end_is_exclusive(self):
        spans = [(0, 3), (3, 7)]
        assert find_span_index(spans, 3) == 1
        assert find_span_index(spans, 7) is None  # end-exclusive

    def test_position_before_all_spans(self):
        spans = [(5, 10), (10, 12)]
        assert find_span_index(spans, 0) is None
        assert find_span_index(spans, 4) is None

    def test_position_after_all_spans(self):
        spans = [(0, 2), (2, 4)]
        assert find_span_index(spans, 4) is None
        assert find_span_index(spans, 100) is None

    def test_first_matching_span_returned_when_overlapping(self):
        spans = [(0, 5), (3, 8)]  # overlapping spans (shouldn't normally happen)
        assert find_span_index(spans, 3) == 0  # first match wins
        assert find_span_index(spans, 6) == 1
