from __future__ import annotations

import pytest

from document_resolution.nlp.detection.structural.structural_reference_compiler import (
    StructuralReferencePatterns,
    compile_structural_reference_patterns,
)


class TestCompileStructuralReferencePatterns:
    @pytest.fixture
    def patterns(self) -> StructuralReferencePatterns:
        return compile_structural_reference_patterns()

    def test_compile_structural_reference_patterns_returns_patterns_container(self, patterns):

        assert isinstance(patterns, StructuralReferencePatterns)

    def test_compile_structural_reference_patterns_matches_schedule_alpha(self, patterns):

        match = patterns.schedule_reference.search("See Schedule A for pricing.")
        assert match is not None
        assert match.group("kind") == "Schedule"
        assert match.group("label") == "A"

    def test_compile_structural_reference_patterns_matches_schedule_numeric(self, patterns):

        match = patterns.schedule_reference.search("See Schedule 1 for pricing.")
        assert match is not None
        assert match.group("kind") == "Schedule"
        assert match.group("label") == "1"

    def test_compile_structural_reference_patterns_matches_exhibit_reference(self, patterns):

        match = patterns.exhibit_reference.search("Attached at Exhibit B.")
        assert match is not None
        assert match.group("kind") == "Exhibit"
        assert match.group("label") == "B"

    def test_compile_structural_reference_patterns_matches_annex_reference(self, patterns):

        match = patterns.annex_reference.search("Further details are in Annex 1.")
        assert match is not None
        assert match.group("kind") == "Annex"
        assert match.group("label") == "1"

    def test_compile_structural_reference_patterns_matches_appendix_reference(self, patterns):

        match = patterns.appendix_reference.search("Refer to Appendix C.")
        assert match is not None
        assert match.group("kind") == "Appendix"
        assert match.group("label") == "C"

    def test_compile_structural_reference_patterns_matches_section_decimal(self, patterns):

        match = patterns.section_reference.search("As set out in Section 4.2 below.")
        assert match is not None
        assert match.group("kind") == "Section"
        assert match.group("label") == "4.2"

    def test_compile_structural_reference_patterns_matches_clause_decimal(self, patterns):

        match = patterns.clause_reference.search("See Clause 7.3 for termination.")
        assert match is not None
        assert match.group("kind") == "Clause"
        assert match.group("label") == "7.3"

    def test_compile_structural_reference_patterns_matches_article_roman(self, patterns):

        match = patterns.article_reference.search("This is governed by Article III.")
        assert match is not None
        assert match.group("kind") == "Article"
        assert match.group("label") == "III"

    def test_compile_structural_reference_patterns_matches_article_numeric(self, patterns):

        match = patterns.article_reference.search("This is governed by Article 2.")
        assert match is not None
        assert match.group("kind") == "Article"
        assert match.group("label") == "2"

    def test_compile_structural_reference_patterns_ignores_plain_capitalised_phrase(self, patterns):

        assert patterns.schedule_reference.search("Annual Budget") is None
        assert patterns.section_reference.search("Board of Directors") is None
        assert patterns.article_reference.search("Company Policy") is None

    def test_compile_structural_reference_patterns_is_case_insensitive_for_kind(self, patterns):

        match = patterns.section_reference.search("see section 4.2 below")
        assert match is not None
        assert match.group("kind") == "section"
        assert match.group("label") == "4.2"

    def test_compile_structural_reference_patterns_matches_appendix_alpha_decimal(self, patterns):

        match = patterns.appendix_reference.search("Refer to Appendix C.13.")
        assert match is not None
        assert match.group("kind") == "Appendix"
        assert match.group("label") == "C.13"

    def test_compile_structural_reference_patterns_matches_appendix_alpha_multi_decimal(self, patterns):

        match = patterns.appendix_reference.search("See Appendix A.1.2 for examples.")
        assert match is not None
        assert match.group("kind") == "Appendix"
        assert match.group("label") == "A.1.2"

    def test_compile_structural_reference_patterns_matches_appendix_reference_before_sentence_punctuation(self,
                                                                                                          patterns):

        match = patterns.appendix_reference.search("Refer to Appendix C. for the schedule.")
        assert match is not None
        assert match.group("kind") == "Appendix"
        assert match.group("label") == "C"
        assert match.group(0) == "Appendix C"
