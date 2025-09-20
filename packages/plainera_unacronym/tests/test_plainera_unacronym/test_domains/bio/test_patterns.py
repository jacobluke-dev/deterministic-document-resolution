from plainera_unacronym.domains.bio.patterns import bio_pattern


class TestBioPattern:

    def test_bio_pattern_prefers_full_rna_token(self):
        pat = bio_pattern()
        m = pat.search("We quantified mRNA levels.")
        assert m and m.group("bio") == "mRNA"

    def test_bio_pattern_other_cases(self):
        pat = bio_pattern()
        assert pat.search("IL-6 increased.").group("bio") == "IL-6"
        assert pat.search("SARS-CoV-2 cohort.").group("bio") == "SARS-CoV-2"
        assert pat.search("the 5′UTR element").group("bio").endswith("UTR")
