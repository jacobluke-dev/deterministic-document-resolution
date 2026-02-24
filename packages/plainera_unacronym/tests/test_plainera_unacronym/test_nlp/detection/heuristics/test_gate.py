import pytest
from plainera_unacronym.nlp.detection.heuristics.gate import (
    CYTOKINE,
    GREEK,
    PCR_RE,
    RNA_RE,
    SECTIONS,
    STATS,
    UNITS,
    VIRUS,
    bio_signal_score,
    should_enable_bio,
)


class TestRegexesMinimalMatches:
    @pytest.mark.parametrize(
        "pattern,sample",
        [
            (RNA_RE, "mRNA expression increased."),
            (CYTOKINE, "IL-10 was elevated."),
            (VIRUS, "H1N1 cases rose."),
            (PCR_RE, "western blot performed."),  # case-insensitive
            (UNITS, "OD600 reached 0.7."),
            (STATS, "odds ratio was significant."),
            (SECTIONS, "Abstract"),
            (GREEK, "contains α only"),
        ],
    )
    def test_minimal_positive(self, pattern, sample):
        assert pattern.search(sample) is not None

    def test_virus_negative_h0n1(self):
        assert VIRUS.search("H0N1 is not valid.") is None

    def test_units_alt_tokens(self):
        assert UNITS.search("10 uL was added.") is not None
        assert UNITS.search("5 ug/mL threshold.") is not None
        assert UNITS.search("OD 260 measured.") is not None

    def test_pcr_variants(self):
        assert PCR_RE.search("PCR was used.") is not None
        assert PCR_RE.search("RT-qPCR confirmed results.") is not None
        assert PCR_RE.search("RTqPCR confirmed results.") is not None
        assert PCR_RE.search("CRISPR/Cas9 editing.") is not None
        assert PCR_RE.search("ELISA validated protein.") is not None


class TestBioSignalScore:
    @pytest.mark.parametrize(
        "text,expected_score,expected_reasons,expected_is_strong",
        [
            ("mRNA present.", 5, {"rna"}, True),  # STRONG
            ("IL-10 increased.", 5, {"cytokine"}, True),  # STRONG
            ("H1N1 detected.", 5, {"virus"}, True),  # STRONG
            ("western blot performed.", 2, {"pcr"}, False),  # SUPPORT
            ("OD600=0.7", 1, {"units"}, False),  # SUPPORT
            ("odds ratio improved.", 2, {"stats"}, False),  # SUPPORT
            ("Abstract", 1, {"sections"}, False),  # SUPPORT
            ("α only", 1, {"greek"}, False),  # SUPPORT
        ],
    )
    def test_minimal_signals(self, text, expected_score, expected_reasons, expected_is_strong):
        score, reasons, is_strong = bio_signal_score(text)
        assert score == expected_score
        assert set(reasons) == expected_reasons
        assert is_strong is expected_is_strong

    def test_strong_plus_support_counts_and_flags(self):
        text = "mRNA present. Abstract mentioned. OD600=0.7"
        score, reasons, is_strong = bio_signal_score(text)
        assert is_strong is True
        assert score == 7  # 5 + 1 + 1
        assert set(reasons) == {"rna", "sections", "units"}

    def test_support_only_does_not_set_is_strong(self):
        text = "PCR performed. Abstract provided."
        score, reasons, is_strong = bio_signal_score(text)
        assert is_strong is False
        assert score == 3  # 2 + 1
        assert set(reasons) == {"pcr", "sections"}

    def test_cytokine_with_greek_counts_both(self):
        text = "TNF-α levels rose."
        score, reasons, is_strong = bio_signal_score(text)
        assert score == 6  # cytokine(5) + greek(1)
        assert set(reasons) == {"cytokine", "greek"}
        assert is_strong is True

    def test_combined_signals(self):
        text = "Abstract. mRNA quantified via RT-qPCR; OD260 noted; 95% CI reported. " "H7N9 was monitored."
        score, reasons, is_strong = bio_signal_score(text)
        # sections(1) + rna(5) + pcr(2) + units(1) + stats(2) + virus(5) = 16
        assert score == 16
        assert set(reasons) == {"sections", "rna", "pcr", "units", "stats", "virus"}
        assert is_strong is True


class TestShouldEnableBio:
    def test_default_false_when_low_signal(self):
        ok, reasons = should_enable_bio("PCR only.")  # score=2
        assert ok is False
        assert set(reasons) == {"pcr"}

    def test_strong_signal_enables_regardless_of_threshold_math(self):
        ok, reasons = should_enable_bio("IL-10")  # score=3
        assert ok is True
        assert set(reasons) == {"cytokine"}

    def test_support_combination_still_false_under_default_threshold(self):
        text = "PCR performed. Abstract provided."  # pcr(2) + sections(1) = 3
        ok, reasons = should_enable_bio(text)  # default threshold=5 → needs ≥8 when no strong
        assert ok is False
        assert set(reasons) == {"pcr", "sections"}

    def test_custom_threshold_does_not_override_strong_requirement(self):
        ok, reasons = should_enable_bio("α only", threshold=1)
        assert ok is False
        assert set(reasons) == {"greek"}

    def test_no_signal(self):
        ok, reasons = should_enable_bio("just narrative prose here")
        assert ok is False
        assert reasons == []
