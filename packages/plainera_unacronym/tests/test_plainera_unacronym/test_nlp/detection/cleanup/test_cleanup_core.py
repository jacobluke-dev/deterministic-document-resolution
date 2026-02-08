from dataclasses import replace

from plainera_unacronym.nlp.detection.cleanup.core import recompute_firsts


class TestRecomputeFirstsUnit:
    def test_uses_existing_normalized_key_without_calling_normalizer(self, cfg, occ, _patch):
        o = occ(cfg, "mRNA", 10, 14)

        # If normalize_acronym_key is called, explode.
        def boom(*args, **kwargs):
            raise AssertionError("normalize_acronym_key should not be called when normalized_key is present")

        _patch(recompute_firsts, normalize_acronym_key=boom)

        firsts = recompute_firsts([o], cfg)

        assert list(firsts.keys()) == [o.normalized_key]
        assert firsts[o.normalized_key].start_offset == 10

    def test_calls_normalizer_when_normalized_key_missing_and_keeps_result(self, cfg, occ, _patch):
        o = occ(cfg, "mRNA", 10, 14)
        o = replace(o, normalized_key="")  # force normalizer path

        calls = {"n": 0}

        def fake_normalize(acr, allow_chars, dotted_mode):
            calls["n"] += 1
            assert acr == "mRNA"
            return "K_MRNA"

        _patch(recompute_firsts, normalize_acronym_key=fake_normalize)

        firsts = recompute_firsts([o], cfg)

        assert calls["n"] == 1
        assert "K_MRNA" in firsts
        assert firsts["K_MRNA"].acronym == "mRNA"

    def test_ignores_occurrence_when_normalizer_returns_empty_key(self, cfg, occ, _patch):
        o = occ(cfg, "mRNA", 10, 14)
        o = replace(o, normalized_key="")  # force normalizer path

        def fake_normalize(*args, **kwargs):
            return ""  # uncomputable key

        _patch(recompute_firsts, normalize_acronym_key=fake_normalize)

        firsts = recompute_firsts([o], cfg)

        assert firsts == {}

    def test_picks_earliest_start_offset_per_key_even_if_confidence_differs(self, cfg, occ, _patch):
        o1 = occ(cfg, "mRNA", 10, 14, conf=0.6)
        o2 = occ(cfg, "mRNA", 2, 6, conf=0.95)

        # Force both to normalise to the same key regardless of input.
        def reminder(acr, allow_chars, dotted_mode):
            return "K"

        _patch(recompute_firsts, normalize_acronym_key=reminder)

        # Force normalizer path for both to keep the unit test strictly about recompute logic.
        o1 = replace(o1, normalized_key="")
        o2 = replace(o2, normalized_key="")

        firsts = recompute_firsts([o1, o2], cfg)

        fo = firsts["K"]
        assert fo.start_offset == 2
        assert fo.end_offset == 6
        assert fo.confidence == 0.95  # from the earliest occurrence chosen


class TestRecomputeFirstsIntegration:
    def test_recompute_firsts_picks_earliest_start_offset_per_key(self, cfg, occ):
        o1 = occ(cfg, "mRNA", 10, 14, conf=0.6)
        o2 = occ(cfg, "mRNA", 2, 6, conf=0.95)

        firsts = recompute_firsts([o1, o2], cfg)

        # Same normalized_key expected for same acronym under same cfg.
        k = o1.normalized_key
        assert k == o2.normalized_key

        fo = firsts[k]
        assert fo.start_offset == 2
        assert fo.end_offset == 6

    def test_recompute_firsts_handles_unsorted_input(self, cfg, occ):
        o_early = occ(cfg, "HTTP", 1, 5)
        o_late = occ(cfg, "HTTP", 30, 34)

        firsts = recompute_firsts([o_late, o_early], cfg)

        fo = firsts[o_early.normalized_key]
        assert fo.start_offset == 1
        assert fo.end_offset == 5

    def test_recompute_firsts_returns_one_entry_per_normalized_key(self, cfg, occ):
        a1 = occ(cfg, "HTTP", 1, 5)
        a2 = occ(cfg, "HTTP", 10, 14)
        b1 = occ(cfg, "RNA", 20, 23)

        firsts = recompute_firsts([a1, a2, b1], cfg)

        assert set(firsts.keys()) == {a1.normalized_key, b1.normalized_key}
        assert firsts[a1.normalized_key].start_offset == 1
        assert firsts[b1.normalized_key].start_offset == 20
