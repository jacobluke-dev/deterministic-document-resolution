from document_resolution.nlp.detection.cleanup.rules.case_typos import (
    _is_alternating_case,
    _is_mixed_case_typo,
    rule_drop_mixed_case_typos,
)


class TestIsAlternatingCase:
    def test_returns_true_for_strict_alternation_len3(self):
        assert _is_alternating_case("aBa") is True
        assert _is_alternating_case("AbA") is True

    def test_returns_true_for_strict_alternation_len4(self):
        assert _is_alternating_case("AbCd") is True
        assert _is_alternating_case("aBcD") is True

    def test_returns_false_when_adjacent_letters_do_not_flip_case(self):
        assert _is_alternating_case("ABcD") is False  # A->B no flip
        assert _is_alternating_case("aBCd") is False  # a->B flip, B->C no flip

    def test_ignores_non_letters(self):
        assert _is_alternating_case("A-bAb") is True  # letters a b A => a b A alternates
        assert _is_alternating_case("a-bAb") is False  # a b A does not alternate (a->b no flip)
        assert _is_alternating_case("A1b2C") is True  # letters A b C alternates
        assert _is_alternating_case("a-B-a") is True  # a B a alternates
        assert _is_alternating_case("A-b-C") is True  # A b C alternates
        assert _is_alternating_case("A1b2C3d") is True  # A b C d alternates

    def test_requires_at_least_three_letters(self):
        assert _is_alternating_case("aB") is False
        assert _is_alternating_case("A- b") is False  # letters A b => only 2 letters

    def test_requires_mixed_case_presence(self):
        assert _is_alternating_case("ABC") is False
        assert _is_alternating_case("abc") is False


class TestIsMixedCaseTypo:
    def test_returns_false_for_short_len_under_4(self):
        assert _is_mixed_case_typo("TfL") is False
        assert _is_mixed_case_typo("aBc") is False
        assert _is_mixed_case_typo("mRNA") is False  # special case letters[0] not upper == False

    def test_flags_internal_single_lowercase_blip(self):
        # Mostly uppercase with one lowercase in the middle and uppercase after -> blip
        assert _is_mixed_case_typo("ABCdE") is True
        assert _is_mixed_case_typo("ABcDE") is True

    def test_does_not_flag_when_lowercase_is_first_letter(self):
        # Allow mRNA/iOS-like prefixes (first letter lowercase)
        assert _is_mixed_case_typo("aBCD") is False
        assert _is_mixed_case_typo("iOSX") is False

    def test_does_not_flag_when_single_lowercase_is_only_at_end(self):
        # No uppercase after the lowercase -> not an internal blip
        assert _is_mixed_case_typo("ABCd") is False

    def test_flags_strict_alternation_at_len_4_or_more(self):
        assert _is_mixed_case_typo("AbCd") is True
        assert _is_mixed_case_typo("aBcD") is True

    def test_does_not_flag_all_upper_or_all_lower(self):
        assert _is_mixed_case_typo("ABCD") is False
        assert _is_mixed_case_typo("abcd") is False


class TestRuleDropMixedCaseTyposUnit:
    def test_drops_internal_case_blip_candidates(self, cfg, occ):
        text = "Noise."

        good = occ(cfg, "HTTP", 0, 4)
        bad = occ(cfg, "ABCdE", 5, 10)  # uppercase after lowercase => blip

        kept, dropped = rule_drop_mixed_case_typos(text, [bad, good])

        assert {o.acronym for o in kept} == {"HTTP"}
        assert any(d.acronym == "ABCdE" and d.rule == "drop_mixed_case_typo" for d in dropped), dropped

    def test_drops_strict_alternation_candidates_len_ge_4(self, cfg, occ):
        text = "Alternation."

        good = occ(cfg, "HTTP", 0, 4)
        bad = occ(cfg, "AbCd", 5, 9)  # strict alternation => typo

        kept, dropped = rule_drop_mixed_case_typos(text, [good, bad])

        assert {o.acronym for o in kept} == {"HTTP"}
        assert any(d.acronym == "AbCd" and d.rule == "drop_mixed_case_typo" for d in dropped), dropped

    def test_does_not_drop_short_mixed_case_len_lt_4(self, cfg, occ):
        text = "Legit short mixed-case."

        legit = occ(cfg, "TfL", 0, 3)

        kept, dropped = rule_drop_mixed_case_typos(text, [legit])

        assert {o.acronym for o in kept} == {"TfL"}
        assert dropped == []

    def test_does_not_drop_mrna_style_prefix_first_letter_lower(self, cfg, occ):
        text = "mRNA style."

        # len=4 letters, but first letter is lower => explicitly allowed
        ok = occ(cfg, "mRNAx", 0, 5)

        kept, dropped = rule_drop_mixed_case_typos(text, [ok])

        assert {o.acronym for o in kept} == {"mRNAx"}
        assert dropped == []

    def test_does_not_drop_single_lower_at_end_without_upper_after(self, cfg, occ):
        text = "Lowercase at end."

        # letters=ABCd => single lowercase is last; there is no uppercase after => not a blip
        ok = occ(cfg, "ABCd", 0, 4)

        kept, dropped = rule_drop_mixed_case_typos(text, [ok])

        assert {o.acronym for o in kept} == {"ABCd"}
        assert dropped == []

    def test_deterministic_sorting_stable_reporting(self, cfg, occ):
        text = "Ordering."

        bad = occ(cfg, "ABCdE", 10, 15)
        good = occ(cfg, "HTTP", 0, 4)

        kept, dropped = rule_drop_mixed_case_typos(text, [bad, good])  # unsorted input

        assert {o.acronym for o in kept} == {"HTTP"}
        assert len(dropped) == 1
        assert dropped[0].acronym == "ABCdE"
        assert dropped[0].rule == "drop_mixed_case_typo"
        assert dropped[0].detail == "mostly_upper_single_lower_or_alternating"
