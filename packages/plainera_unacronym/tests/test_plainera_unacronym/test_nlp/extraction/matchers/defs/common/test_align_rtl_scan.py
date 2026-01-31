from plainera_unacronym.nlp.extraction.matchers.defs.common import _align_rtl_scan, InitialsStream, AlignmentHit


class TestAlignRtlScanWrapper:
    def test_returns_none_when_align_rtl_scan_returns_none(self, _patch):
        _patch(_align_rtl_scan, align_rtl_scan=lambda *a, **k: None)
        stream = InitialsStream(letters=["A"], owners=[0], is_stop=[False])
        assert _align_rtl_scan(
            ["A"],
            stream=stream,
            allow_upper_on_stop=False,
            allow_lower_on_non_stop=False,
        ) is None

    def test_builds_alignment_hit_from_used_positions(self, _patch):
        # pretend positions 0 and 2 were used
        _patch(_align_rtl_scan, align_rtl_scan=lambda *a, **k: [0, 2])

        stream = InitialsStream(
            letters=["A", "B", "C"],
            owners=[2, 1, 2],  # positions 0 and 2 both owned by token 2
            is_stop=[False, False, False],
        )

        hit = _align_rtl_scan(
            ["A", "C"],
            stream=stream,
            allow_upper_on_stop=False,
            allow_lower_on_non_stop=False,
        )

        assert isinstance(hit, AlignmentHit)
        assert hit.used_letter_pos == [0, 2]
        assert hit.hit_tokens == {2}
        assert hit.tok_left == 2
        assert hit.tok_right == 2

    def test_tok_bounds_span_multiple_tokens(self, _patch):
        _patch(_align_rtl_scan, align_rtl_scan=lambda *a, **k: [0, 1])

        stream = InitialsStream(
            letters=["A", "B"],
            owners=[3, 1],
            is_stop=[False, False],
        )

        hit = _align_rtl_scan(
            ["A", "B"],
            stream=stream,
            allow_upper_on_stop=False,
            allow_lower_on_non_stop=False,
        )

        assert hit.hit_tokens == {1, 3}
        assert hit.tok_left == 1
        assert hit.tok_right == 3
