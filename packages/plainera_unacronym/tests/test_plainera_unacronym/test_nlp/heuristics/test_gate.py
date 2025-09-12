import pytest

from plainera_unacronym.nlp.heuristics.gate import _slice, RNA_RE, CYTOKINE, PCR_RE, UNITS, STATS, VIRUS, SECTIONS, \
    GREEK, bio_signal_score, should_enable_bio


@pytest.mark.unit
class TestSlice:
    def test_slice_truncates_when_over_max(self):
        text = "x" * 100
        assert _slice(text, max_chars=10) == "x" * 10

    def test_slice_noop_when_shorter_or_equal(self):
        text = "hello"
        assert _slice(text, max_chars=10) == "hello"
        assert _slice(text, max_chars=5) == "hello"


@pytest.mark.unit
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


@pytest.mark.unit
class TestBioSignalScore:
    @pytest.mark.parametrize(
        "text,expected_score,expected_reasons",
        [
            ("mRNA present.", 3, {"rna"}),
            ("IL-10 increased.", 3, {"cytokine"}),
            ("H1N1 detected.", 3, {"virus"}),
            ("western blot performed.", 2, {"pcr"}),
            ("OD600=0.7", 1, {"units"}),
            ("odds ratio improved.", 2, {"stats"}),
            ("Abstract", 1, {"sections"}),
            ("α only", 1, {"greek"}),
        ],
    )
    def test_minimal_signals(self, text, expected_score, expected_reasons):
        score, reasons = bio_signal_score(text)
        assert score == expected_score
        assert set(reasons) == expected_reasons

    def test_cytokine_with_greek_counts_both(self):
        text = "TNF-α levels rose."
        score, reasons = bio_signal_score(text)
        # cytokine (3) + greek (1)
        assert score == 4
        assert set(reasons) == {"cytokine", "greek"}

    def test_combined_signals(self):
        text = (
            "Abstract. mRNA quantified via RT-qPCR; OD260 noted; 95% CI reported. "
            "H7N9 was monitored."
        )
        score, reasons = bio_signal_score(text)
        # sections(1) + rna(3) + pcr(2) + units(1) + stats(2) + virus(3) = 12
        assert score == 12
        assert set(reasons) == {"sections", "rna", "pcr", "units", "stats", "virus"}


@pytest.mark.unit
class TestShouldEnableBio:
    def test_threshold_default_false_when_low_signal(self):
        ok, reasons = should_enable_bio("PCR only.")  # score=2
        assert ok is False
        assert set(reasons) == {"pcr"}

    def test_threshold_default_true_when_score_meets_or_exceeds(self):
        ok, reasons = should_enable_bio("IL-10")  # score=3
        assert ok is True
        assert set(reasons) == {"cytokine"}

    def test_combination_reaches_threshold(self):
        text = "PCR performed. Abstract provided."  # pcr(2) + sections(1) = 3
        ok, reasons = should_enable_bio(text)
        assert ok is True
        assert set(reasons) == {"pcr", "sections"}

    def test_custom_threshold(self):
        ok, reasons = should_enable_bio("α only", threshold=1)  # greek(1)
        assert ok is True
        assert set(reasons) == {"greek"}

    def test_no_signal(self):
        ok, reasons = should_enable_bio("just narrative prose here")
        assert ok is False
        assert reasons == []
