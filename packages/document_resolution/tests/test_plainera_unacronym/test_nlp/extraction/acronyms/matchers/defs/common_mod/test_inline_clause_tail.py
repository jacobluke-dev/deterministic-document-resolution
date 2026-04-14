from document_resolution.nlp.extraction.acronyms.matchers.defs.common import inline_clause_tail


class TestInlineClauseTail:
    def test_returns_full_string_when_no_boundary(self):
        s = "Alpha Beta Gamma"
        tail, end = inline_clause_tail(s)
        assert tail == s
        assert end == len(s)

    def test_stops_at_boundary_dot_when_followed_by_space(self):
        s = "Alpha. Beta"
        tail, end = inline_clause_tail(s)
        assert tail == "Alpha"
        assert end == len("Alpha")

    def test_does_not_stop_at_dot_when_followed_by_letter(self):
        # boundary regex is [.;:](?=\\s|$) so "e.g." should not stop at the first dot
        s = "e.g. Alpha"
        tail, end = inline_clause_tail(s)
        # first dot is followed by 'g' (no stop), second dot followed by space (stop at that dot)
        assert tail == "e.g"
        assert end == len("e.g")

    def test_stops_at_colon_or_semicolon_when_followed_by_space(self):
        s = "Alpha: Beta"
        tail, end = inline_clause_tail(s)
        assert tail == "Alpha"
        assert end == len("Alpha")

        s2 = "Alpha; Beta"
        tail2, end2 = inline_clause_tail(s2)
        assert tail2 == "Alpha"
        assert end2 == len("Alpha")

    def test_stops_on_newline(self):
        s = "Alpha\nBeta"
        tail, end = inline_clause_tail(s)
        assert tail == "Alpha"
        assert end == len("Alpha")
