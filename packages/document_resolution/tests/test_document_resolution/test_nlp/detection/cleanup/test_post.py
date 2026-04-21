from document_resolution.nlp.common.types import AcronymDetectorResult
from document_resolution.nlp.detection.cleanup.post import post_detect_cleanup


class TestPostCleanup:
    def test_cleanup_drops_rna_inside_mrna_by_contained_suffix(self, cfg, occ, fo):
        text = "We measured mRNA expression."

        occ_mrna = occ(cfg, "mRNA", 12, 16)
        occ_rna = occ(cfg, "RNA", 13, 16)  # contained, strict suffix

        det = AcronymDetectorResult(
            unique_acronyms={
                occ_mrna.normalized_key: fo("mRNA", 12, 16),
                occ_rna.normalized_key: fo("RNA", 13, 16),
            },
            occurrences=[occ_mrna, occ_rna],
        )

        cleaned, summary, dropped = post_detect_cleanup(text, det, cfg)

        assert len(cleaned.occurrences) == 1
        assert cleaned.occurrences[0].acronym == "mRNA"
        assert "RNA" not in {fo.acronym for fo in cleaned.unique_acronyms.values()}
        assert any(d.acronym == "RNA" and d.rule == "contained_suffix" for d in dropped), dropped

    def test_cleanup_does_not_drop_when_not_suffix_even_if_contained(self, cfg, occ, fo):
        text = "Token ABCD appears."

        # Contained span, but not suffix relationship -> should be retained
        occ_outer = occ(cfg, "ABCD", 6, 10)
        occ_inner = occ(cfg, "BC", 7, 9)  # contained, but "ABCD" does not end with "BC"? (it ends with "CD")

        det = AcronymDetectorResult(
            unique_acronyms={
                occ_outer.normalized_key: fo("ABCD", 6, 10),
                occ_inner.normalized_key: fo("BC", 7, 9),
            },
            occurrences=[occ_outer, occ_inner],
        )

        cleaned, _, dropped = post_detect_cleanup(text, det, cfg)

        assert {o.acronym for o in cleaned.occurrences} == {"ABCD", "BC"}
        assert dropped == []

    def test_cleanup_drops_token_before_paren_suffix_messenger_rna_mrna(self, cfg, occ, fo):
        text = "messenger RNA (mRNA) has been developed,"

        occ_rna = occ(cfg, "RNA", 10, 13, conf=0.6)
        occ_mrna = occ(cfg, "mRNA", 15, 19, conf=0.85)

        det = AcronymDetectorResult(
            unique_acronyms={
                occ_rna.normalized_key: fo("RNA", 10, 13, conf=0.6),
                occ_mrna.normalized_key: fo("mRNA", 15, 19, conf=0.85),
            },
            occurrences=[occ_rna, occ_mrna],
        )

        cleaned, _, dropped = post_detect_cleanup(text, det, cfg)

        assert {o.acronym for o in cleaned.occurrences} == {"mRNA"}
        assert "RNA" not in {fo.acronym for fo in cleaned.unique_acronyms.values()}
        assert any(d.acronym == "RNA" and d.rule == "token_before_paren_suffix" for d in dropped), dropped

    def test_cleanup_drops_rna_inside_parens_after_left_mrna(self, cfg, occ, fo):
        text = "We measured mRNA (RNA) expression."

        # mRNA followed by parens containing RNA; drop RNA inside parens.
        occ_mrna = occ(cfg, "mRNA", 12, 16)
        occ_rna = occ(cfg, "RNA", 18, 21)

        det = AcronymDetectorResult(
            unique_acronyms={
                occ_mrna.normalized_key: fo("mRNA", 12, 16),
                occ_rna.normalized_key: fo("RNA", 18, 21),
            },
            occurrences=[occ_mrna, occ_rna],
        )

        cleaned, _, dropped = post_detect_cleanup(text, det, cfg)

        assert {o.acronym for o in cleaned.occurrences} == {"mRNA"}
        assert any(d.acronym == "RNA" and d.rule == "inside_paren_suffix_of_left" for d in dropped), dropped

    def test_cleanup_does_not_drop_mixed_inside_parens_if_not_allcaps(self, cfg, occ, fo):
        text = "We measured mRNA (rNa) expression."

        occ_mrna = occ(cfg, "mRNA", 12, 16)
        occ_rna = occ(cfg, "rNa", 18, 21)

        det = AcronymDetectorResult(
            unique_acronyms={
                occ_mrna.normalized_key: fo("mRNA", 12, 16),
                occ_rna.normalized_key: fo("rNa", 18, 21),
            },
            occurrences=[occ_mrna, occ_rna],
        )

        cleaned, _, dropped = post_detect_cleanup(text, det, cfg)

        # inside_paren_suffix_of_left is intentionally narrow: ALLCAPS only.
        assert {o.acronym for o in cleaned.occurrences} == {"mRNA", "rNa"}
        assert dropped == []

    def test_cleanup_drops_shorter_suffix_when_same_end_offset_end_suffix_micro(self, cfg, occ, fo):
        text = "We measured mRNA expression."

        # Same end offset (16), different starts. Shorter "RNA" is suffix of "mRNA" => drop RNA.
        occ_mrna = occ(cfg, "mRNA", 12, 16)
        occ_rna = occ(cfg, "RNA", 13, 16)

        det = AcronymDetectorResult(
            unique_acronyms={
                occ_mrna.normalized_key: fo("mRNA", 12, 16),
                occ_rna.normalized_key: fo("RNA", 13, 16),
            },
            occurrences=[occ_mrna, occ_rna],
        )

        cleaned, _, dropped = post_detect_cleanup(text, det, cfg)

        assert {o.acronym for o in cleaned.occurrences} == {"mRNA"}
        # Earlier rules may claim the drop before end_suffix_micro runs.
        assert any(d.acronym == "RNA" and d.rule in {"contained_suffix", "end_suffix_micro"} for d in dropped), dropped

    def test_cleanup_drops_mixed_case_typo_internal_blip(self, cfg, occ, fo):
        text = "We measured ABCdE levels."

        # letters=ABCDE with one lowercase 'd' in the middle -> should drop (len>=4, mostly upper, blip)
        occ_typo = occ(cfg, "ABCdE", 12, 17)

        det = AcronymDetectorResult(
            unique_acronyms={
                occ_typo.normalized_key: fo("ABCdE", 12, 17),
            },
            occurrences=[occ_typo],
        )

        cleaned, _, dropped = post_detect_cleanup(text, det, cfg)

        assert cleaned.occurrences == []
        assert cleaned.unique_acronyms == {}
        assert any(d.acronym == "ABCdE" and d.rule == "drop_mixed_case_typo" for d in dropped), dropped

    def test_cleanup_does_not_drop_known_3char_mixed_case_like_tfl(self, cfg, occ, fo):
        text = "We travelled via TfL today."

        occ_ok = occ(cfg, "TfL", 17, 20)

        det = AcronymDetectorResult(
            unique_acronyms={occ_ok.normalized_key: fo("TfL", 17, 20)},
            occurrences=[occ_ok],
        )

        cleaned, _, dropped = post_detect_cleanup(text, det, cfg)

        assert {o.acronym for o in cleaned.occurrences} == {"TfL"}
        assert dropped == []

    def test_cleanup_recomputes_firsts_from_kept_occurrences_earliest_wins(self, cfg, occ, fo):
        text = "mRNA appears. Later, mRNA appears again."

        occ1 = occ(cfg, "mRNA", 0, 4, conf=0.7)
        occ2 = occ(cfg, "mRNA", 22, 26, conf=0.9)

        det = AcronymDetectorResult(
            unique_acronyms={
                occ2.normalized_key: fo("mRNA", 22, 26, conf=0.9),  # pretend detector picked later first
            },
            occurrences=[occ2, occ1],  # unsorted input on purpose
        )

        cleaned, _, _ = post_detect_cleanup(text, det, cfg)

        # recompute_firsts should select the earliest start_offset regardless of input ordering
        fo = next(iter(cleaned.unique_acronyms.values()))
        assert fo.acronym == "mRNA"
        assert fo.start_offset == 0
        assert fo.end_offset == 4
