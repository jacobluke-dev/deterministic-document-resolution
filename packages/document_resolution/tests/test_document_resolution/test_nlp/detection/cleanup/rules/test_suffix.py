from document_resolution.nlp.detection.cleanup.rules.suffix import (
    _is_strict_suffix,
    rule_contained_suffix,
    rule_end_suffix_micro,
    rule_inside_paren_suffix_of_left_acronym,
    rule_token_before_paren_suffix,
)


class TestIsStrictSuffix:
    def test_returns_true_for_case_insensitive_suffix(self):
        assert _is_strict_suffix("RNA", "mRNA") is True
        assert _is_strict_suffix("rna", "mRNA") is True
        assert _is_strict_suffix("RnA", "mRNA") is True

    def test_returns_false_when_equal_length(self):
        assert _is_strict_suffix("mRNA", "mRNA") is False
        assert _is_strict_suffix("RNA", "RNA") is False

    def test_returns_false_when_not_a_suffix(self):
        assert _is_strict_suffix("BC", "ABCD") is False
        assert _is_strict_suffix("MR", "mRNA") is False

    def test_returns_false_when_shorter_is_longer_or_equal(self):
        assert _is_strict_suffix("ABCD", "ABC") is False
        assert _is_strict_suffix("ABC", "ABC") is False

    def test_empty_string_behaviour_is_documented(self):
        # Strict suffix: empty string is strictly shorter than any non-empty string.
        assert _is_strict_suffix("", "mRNA") is True
        # Not strict when both empty (equal length)
        assert _is_strict_suffix("", "") is False


class TestRuleContainedSuffixUnit:
    def test_drops_inner_when_strictly_contained_and_suffix(self, cfg, occ):
        text = "We measured mRNA expression."

        outer = occ(cfg, "mRNA", 12, 16)
        inner = occ(cfg, "RNA", 13, 16)  # contained + suffix

        kept, dropped = rule_contained_suffix(text, [outer, inner])

        assert {o.acronym for o in kept} == {"mRNA"}
        assert any(d.acronym == "RNA" and d.rule == "contained_suffix" for d in dropped), dropped

    def test_does_not_drop_when_contained_but_not_suffix(self, cfg, occ):
        text = "Token ABCD appears."

        outer = occ(cfg, "ABCD", 6, 10)
        inner = occ(cfg, "BC", 7, 9)  # contained, but not suffix of ABCD

        kept, dropped = rule_contained_suffix(text, [outer, inner])

        assert {o.acronym for o in kept} == {"ABCD", "BC"}
        assert dropped == []

    def test_does_not_drop_when_same_span_exact_duplicate(self, cfg, occ):
        text = "Duplicate span."

        a = occ(cfg, "mRNA", 0, 4)
        b = occ(cfg, "RNA", 0, 4)  # same offsets; rule explicitly skips exact span matches

        kept, dropped = rule_contained_suffix(text, [a, b])

        assert {(o.acronym, o.start_offset, o.end_offset) for o in kept} == {
            ("mRNA", 0, 4),
            ("RNA", 0, 4),
        }
        assert dropped == []

    def test_deterministic_drop_reporting_is_stable(self, cfg, occ):
        text = "We measured mRNA expression."

        # Give out-of-order input on purpose
        inner = occ(cfg, "RNA", 13, 16)
        outer = occ(cfg, "mRNA", 12, 16)

        kept, dropped = rule_contained_suffix(text, [inner, outer])

        assert {o.acronym for o in kept} == {"mRNA"}
        assert len(dropped) == 1
        assert dropped[0].rule == "contained_suffix"
        assert dropped[0].detail.startswith("contained_in=mRNA@(")

    def test_does_not_drop_when_inner_is_not_strictly_shorter(self, cfg, occ):
        text = "Equal length tokens."

        outer = occ(cfg, "RNA", 10, 13)
        inner = occ(cfg, "rna", 10, 13)  # same length; _is_strict_suffix must return False

        kept, dropped = rule_contained_suffix(text, [outer, inner])

        assert {o.acronym for o in kept} == {"RNA", "rna"}
        assert dropped == []


class TestRuleEndSuffixMicroUnit:
    def test_drops_shorter_suffix_with_same_end_offset(self, cfg, occ):
        text = "We measured mRNA expression."

        long = occ(cfg, "mRNA", 12, 16)
        short = occ(cfg, "RNA", 13, 16)  # same end, strict suffix

        kept, dropped = rule_end_suffix_micro(text, [short, long])  # unsorted input

        assert {o.acronym for o in kept} == {"mRNA"}
        assert any(d.acronym == "RNA" and d.rule == "end_suffix_micro" for d in dropped), dropped

    def test_does_not_drop_when_same_end_but_not_suffix(self, cfg, occ):
        text = "Token ABCD appears."

        long = occ(cfg, "ABCD", 6, 10)
        short = occ(cfg, "AB", 8, 10)  # same end, but "ABCD" does not end with "AB"

        kept, dropped = rule_end_suffix_micro(text, [long, short])

        assert {o.acronym for o in kept} == {"ABCD", "AB"}
        assert dropped == []

    def test_only_compares_within_same_end_offset_group(self, cfg, occ):
        text = "Two groups."

        # Group end=10
        a1 = occ(cfg, "mRNA", 6, 10)
        a2 = occ(cfg, "RNA", 7, 10)  # would drop within group

        # Group end=20
        b1 = occ(cfg, "HTTP", 16, 20)
        b2 = occ(cfg, "HT", 18, 20)  # NOT a suffix of HTTP

        kept, dropped = rule_end_suffix_micro(text, [b2, a2, b1, a1])

        assert {o.acronym for o in kept} == {"mRNA", "HTTP", "HT"}
        assert any(d.acronym == "RNA" and d.rule == "end_suffix_micro" for d in dropped), dropped

    def test_drops_intermediate_suffixes_when_multiple_candidates(self, cfg, occ):
        text = "Multiple candidates."

        c1 = occ(cfg, "xmRNA", 5, 10)  # longest
        c2 = occ(cfg, "mRNA", 6, 10)  # suffix of xmRNA -> will drop
        c3 = occ(cfg, "RNA", 7, 10)  # suffix of both -> will drop

        kept, dropped = rule_end_suffix_micro(text, [c3, c2, c1])

        assert {o.acronym for o in kept} == {"xmRNA"}
        assert any(d.acronym == "mRNA" and d.rule == "end_suffix_micro" for d in dropped), dropped
        assert any(d.acronym == "RNA" and d.rule == "end_suffix_micro" for d in dropped), dropped


class TestRuleInsideParenSuffixOfLeftUnit:
    def test_drops_allcaps_inner_when_suffix_of_left(self, cfg, occ):
        text = "We measured mRNA (RNA) expression."

        left = occ(cfg, "mRNA", 12, 16)
        inner = occ(cfg, "RNA", 18, 21)

        kept, dropped = rule_inside_paren_suffix_of_left_acronym(text, [left, inner])

        assert {o.acronym for o in kept} == {"mRNA"}
        assert any(d.acronym == "RNA" and d.rule == "inside_paren_suffix_of_left" for d in dropped), dropped

    def test_does_not_drop_when_left_not_followed_by_paren(self, cfg, occ):
        text = "We measured mRNA RNA expression."

        left = occ(cfg, "mRNA", 12, 16)
        inner = occ(cfg, "RNA", 17, 20)  # not in parentheses

        kept, dropped = rule_inside_paren_suffix_of_left_acronym(text, [left, inner])

        assert {o.acronym for o in kept} == {"mRNA", "RNA"}
        assert dropped == []

    def test_does_not_drop_when_inner_not_allcaps_after_trim(self, cfg, occ):
        text = "We measured mRNA (rNa) expression."

        left = occ(cfg, "mRNA", 12, 16)
        inner = occ(cfg, "rNa", 18, 21)

        kept, dropped = rule_inside_paren_suffix_of_left_acronym(text, [left, inner])

        assert {o.acronym for o in kept} == {"mRNA", "rNa"}
        assert dropped == []

    def test_does_not_drop_when_inner_is_not_suffix(self, cfg, occ):
        text = "We measured ABCD (BC) expression."

        left = occ(cfg, "ABCD", 12, 16)
        inner = occ(cfg, "BC", 18, 20)  # not suffix of ABCD (endswith CD)

        kept, dropped = rule_inside_paren_suffix_of_left_acronym(text, [left, inner])

        assert {o.acronym for o in kept} == {"ABCD", "BC"}
        assert dropped == []

    def test_inner_must_be_within_first_closing_paren_after_left(self, cfg, occ):
        text = "We measured mRNA (RNA) and (RNA) again."

        left = occ(cfg, "mRNA", 12, 16)
        inner1 = occ(cfg, "RNA", 18, 21)  # inside first parens -> should drop
        inner2 = occ(cfg, "RNA", 28, 31)  # in later parens -> should NOT be dropped by left match

        kept, dropped = rule_inside_paren_suffix_of_left_acronym(text, [left, inner2, inner1])

        assert {o.acronym for o in kept} == {"mRNA", "RNA"}  # inner2 remains
        assert any(d.start == 18 and d.end == 21 and d.rule == "inside_paren_suffix_of_left" for d in dropped), dropped

    def test_trims_punct_on_inner_before_allcaps_check_and_suffix_match(self, cfg, occ):
        text = "We measured mRNA (RNA,) expression."

        left = occ(cfg, "mRNA", 12, 16)
        # Inner acronym includes punctuation that should be stripped by PUNCT_TRIM (comma)
        inner = occ(cfg, "RNA,", 18, 22)

        kept, dropped = rule_inside_paren_suffix_of_left_acronym(text, [left, inner])

        assert {o.acronym for o in kept} == {"mRNA"}
        assert any(d.acronym == "RNA," and d.rule == "inside_paren_suffix_of_left" for d in dropped), dropped


class TestRuleTokenBeforeParenSuffix:
    def test_drops_allcaps_token_before_paren_when_suffix_matches(self, cfg, occ):
        text = "messenger RNA (mRNA) has been developed,"

        a = occ(cfg, "RNA", 10, 13, conf=0.6)
        b = occ(cfg, "mRNA", 15, 19, conf=0.85)

        kept, dropped = rule_token_before_paren_suffix(text, [a, b])

        assert {o.acronym for o in kept} == {"mRNA"}
        assert any(d.acronym == "RNA" and d.rule == "token_before_paren_suffix" for d in dropped), dropped

    def test_does_not_drop_when_token_not_allcaps(self, cfg, occ):
        text = "messenger rna (mRNA) has been developed,"

        a = occ(cfg, "rna", 10, 13, conf=0.6)  # not ALLCAPS
        b = occ(cfg, "mRNA", 15, 19, conf=0.85)

        kept, dropped = rule_token_before_paren_suffix(text, [a, b])

        assert {o.acronym for o in kept} == {"rna", "mRNA"}
        assert dropped == []

    def test_does_not_drop_when_paren_not_immediately_closed_after_b(self, cfg, occ):
        text = "messenger RNA (mRNA extra) has been developed,"

        a = occ(cfg, "RNA", 10, 13)
        b = occ(cfg, "mRNA", 15, 19)

        kept, dropped = rule_token_before_paren_suffix(text, [a, b])

        # B is not immediately followed by ')', so rule should not fire.
        assert {o.acronym for o in kept} == {"RNA", "mRNA"}
        assert dropped == []

    def test_respects_max_ws_between_a_and_paren(self, cfg, occ):
        text = "messenger RNA   (mRNA) has been developed,"

        a = occ(cfg, "RNA", 10, 13)
        b = occ(cfg, "mRNA", 17, 21)

        # default max_ws=2; there are 3 spaces before '(' -> should not drop
        kept, dropped = rule_token_before_paren_suffix(text, [a, b])
        assert {o.acronym for o in kept} == {"RNA", "mRNA"}
        assert dropped == []

        # allow more whitespace -> should drop
        kept2, dropped2 = rule_token_before_paren_suffix(text, [a, b], max_ws=3)
        assert {o.acronym for o in kept2} == {"mRNA"}
        assert any(d.rule == "token_before_paren_suffix" for d in dropped2), dropped2

    def test_chooses_best_b_when_multiple_candidates_start_at_same_offset(self, cfg, occ):
        text = "messenger RNA (mRNA) has been developed,"

        a = occ(cfg, "RNA", 10, 13, conf=0.6)

        # Two candidates starting at same offset inside parens:
        # pick the longest first (then confidence), so "mRNA" beats "mR".
        b_short = occ(cfg, "mR", 15, 17, conf=0.99)
        b_long = occ(cfg, "mRNA", 15, 19, conf=0.80)

        kept, dropped = rule_token_before_paren_suffix(text, [a, b_short, b_long])

        assert {o.acronym for o in kept} == {"mR", "mRNA"} or {o.acronym for o in kept} == {"mRNA", "mR"}
        # Token A should be dropped based on best B="mRNA"
        assert any(d.acronym == "RNA" and d.rule == "token_before_paren_suffix" for d in dropped), dropped
