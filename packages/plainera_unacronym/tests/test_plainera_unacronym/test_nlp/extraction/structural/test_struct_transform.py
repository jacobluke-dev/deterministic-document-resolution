import pytest
from plainera_unacronym.nlp.detection.structural import StructuralReference
from plainera_unacronym.nlp.extraction.structural.config import StructuralReferenceExtractionConfig
from plainera_unacronym.nlp.extraction.structural.transform import _canonicalize_structural_reference, roman_to_int


class TestCanonicalizeStructuralReference:
    def test_canonicalize_structural_reference_passthrough_when_disabled(self):
        ref = StructuralReference(
            kind="Article",
            label="III",
            start_offset=0,
            end_offset=11,
            normalized_key="article_iii",
            provenance="structural_reference_detector",
        )
        cfg = StructuralReferenceExtractionConfig(convert_roman_numerals=False)

        canonical_label, canonical_key = _canonicalize_structural_reference(ref, cfg)

        assert canonical_label == "III"
        assert canonical_key == "article_iii"

    def test_canonicalize_structural_reference_converts_article_roman_when_enabled(self):
        ref = StructuralReference(
            kind="Article",
            label="III",
            start_offset=0,
            end_offset=11,
            normalized_key="article_iii",
            provenance="structural_reference_detector",
        )
        cfg = StructuralReferenceExtractionConfig(convert_roman_numerals=True)

        canonical_label, canonical_key = _canonicalize_structural_reference(ref, cfg)

        assert canonical_label == "3"
        assert canonical_key == "article_3"

    def test_canonicalize_structural_reference_does_not_convert_non_article_roman(self):
        ref = StructuralReference(
            kind="Appendix",
            label="III",
            start_offset=0,
            end_offset=12,
            normalized_key="appendix_iii",
            provenance="structural_reference_detector",
        )
        cfg = StructuralReferenceExtractionConfig(convert_roman_numerals=True)

        canonical_label, canonical_key = _canonicalize_structural_reference(ref, cfg)

        assert canonical_label == "III"
        assert canonical_key == "appendix_iii"
