from document_resolution.nlp.extraction.acronyms.matchers.defs import find_inline_longform_after_acr


class TestFindInlineLongformAfterAcrIntegrationFastPath:
    def test_empty_snippet_returns_empty(self, dummy_cfg):
        cfg = dummy_cfg()
        assert find_inline_longform_after_acr("", cfg, acr="PDF", require_initials_match=False) == []

    def test_fast_path_takes_up_to_6_tokens(self, dummy_cfg):
        cfg = dummy_cfg(max_phrase_chars=200)
        snippet = "alpha beta gamma delta epsilon zeta eta theta"
        out = find_inline_longform_after_acr(snippet, cfg, acr="X", require_initials_match=False)
        assert len(out) == 1
        m = out[0]
        assert m.definition == "alpha beta gamma delta epsilon zeta"
        assert snippet[m.def_start : m.def_end] == "alpha beta gamma delta epsilon zeta"
        assert m.raw == "alpha beta gamma delta epsilon zeta"

    def test_fast_path_stops_on_clause_boundary_token(self, dummy_cfg):
        cfg = dummy_cfg(max_phrase_chars=200)
        snippet = "alpha beta. gamma delta"
        out = find_inline_longform_after_acr(snippet, cfg, acr="X", require_initials_match=False)
        assert len(out) == 1
        m = out[0]
        assert snippet[m.def_start : m.def_end] == "alpha beta."
        assert m.raw == "alpha beta."
        assert m.definition == "alpha beta"

    def test_fast_path_gates_raw_window_length(self, dummy_cfg):
        cfg = dummy_cfg(max_phrase_chars=5)
        snippet = "a b c"
        # raw_window == "a b c" (len 5) OK
        out = find_inline_longform_after_acr(snippet, cfg, acr="X", require_initials_match=False)
        assert len(out) == 1

        snippet2 = "a b c d"
        # raw_window == "a b c d" (len 7) > max → blocked
        assert find_inline_longform_after_acr(snippet2, cfg, acr="X", require_initials_match=False) == []


class TestFindInlineLongformAfterAcrUnitGatesAndCue:
    def test_global_tail_gate_blocks(self, _patch, dummy_cfg):
        cfg = dummy_cfg(max_phrase_chars=10)

        calls = {"n": 0}

        def fake_inline_clause_tail(_s):
            calls["n"] += 1
            # first call: code does collapse_ws(tail[0]) so tail must be indexable
            if calls["n"] == 1:
                return ["X" * 999], 0
            # second call: code does collapse_ws(tail) so tail must be a string
            return "X" * 999, 0

        _patch(
            find_inline_longform_after_acr,
            inline_clause_tail=fake_inline_clause_tail,
            collapse_ws=lambda s: s,
        )

        assert find_inline_longform_after_acr("anything", cfg, acr="PDF", require_initials_match=False) == []

    def test_require_initials_match_requires_cue_prefix(self, _patch, dummy_cfg):
        cfg = dummy_cfg(max_phrase_chars=200)

        def fake_inline_clause_tail(s):
            # keep gates happy
            return (["ok"], 0) if "FIRST" not in getattr(fake_inline_clause_tail, "_seen", set()) else ("ok", 0)

        # simpler: stateful
        state = {"n": 0}

        def fake_inline_clause_tail2(_s):
            state["n"] += 1
            return (["ok"], 0) if state["n"] == 1 else ("ok", 0)

        _patch(
            find_inline_longform_after_acr,
            inline_clause_tail=fake_inline_clause_tail2,
            collapse_ws=lambda s: s,
            strip_inline_cue_prefix=lambda s, cfg: None,  # no cue => not our pattern
        )

        assert (
            find_inline_longform_after_acr(" - Portable Document Format", cfg, acr="PDF", require_initials_match=True)
            == []
        )


class TestFindInlineLongformAfterAcrUnitAlignmentAndFallback:
    def test_align_fallback_called_when_strict_fails(self, _patch, dummy_cfg, hit_cfg, build_stream_seen):
        cfg = dummy_cfg(max_phrase_chars=200)
        build_stream_fn, _seen = build_stream_seen

        calls = {"allow_upper_on_stop": []}

        state = {"n": 0}

        def fake_inline_clause_tail(_s):
            state["n"] += 1
            return (["ok"], 0) if state["n"] == 1 else ("ok", 0)

        def fake_strip_inline_cue_prefix(s, cfg):
            return "Portable Document Format", 0

        def fake_align(acr, stream, tokens, **kwargs):
            calls["allow_upper_on_stop"].append(kwargs.get("allow_upper_on_stop"))
            if kwargs.get("allow_upper_on_stop") is False:
                return None
            return hit_cfg(tok_left=0, tok_right=2, hit_tokens={0, 1, 2})

        def fake_kept_token_indices(tokens, **kwargs):
            return [0, 1, 2]

        def fake_phrase_from_indices(tokens, idxs):
            return " ".join(tokens[i] for i in idxs)

        _patch(
            find_inline_longform_after_acr,
            inline_clause_tail=fake_inline_clause_tail,
            collapse_ws=lambda s: " ".join(str(s).split()),
            strip_inline_cue_prefix=fake_strip_inline_cue_prefix,
            build_initials_stream=build_stream_fn,
            align_acronym_to_initials=fake_align,
            kept_token_indices=fake_kept_token_indices,
            phrase_from_indices=fake_phrase_from_indices,
            tighten_definition_span=lambda s: s,
            normalize_definition=lambda s: s,
            is_mixed_case_acronym=lambda acr: False,
            first_alnum_char_upper=lambda tok: tok[0].upper() if tok else None,
        )

        out = find_inline_longform_after_acr("Portable Document Format", cfg, acr="PDF", require_initials_match=True)
        assert len(out) == 1
        assert calls["allow_upper_on_stop"] == [False, True]

    def test_build_initials_stream_called_with_expected_flags(self, _patch, dummy_cfg, hit_cfg, build_stream_seen):
        cfg = dummy_cfg(max_phrase_chars=200)
        build_stream_fn, seen = build_stream_seen

        state = {"n": 0}

        def fake_inline_clause_tail(_s):
            state["n"] += 1
            return (["ok"], 0) if state["n"] == 1 else ("ok", 0)

        def fake_strip_inline_cue_prefix(s, cfg):
            return ("Portable Document Format", 0)

        def fake_align(acr, stream, tokens, **kwargs):
            return hit_cfg(tok_left=0, tok_right=2, hit_tokens={0, 1, 2})

        _patch(
            find_inline_longform_after_acr,
            inline_clause_tail=fake_inline_clause_tail,
            collapse_ws=lambda s: " ".join(str(s).split()),
            strip_inline_cue_prefix=fake_strip_inline_cue_prefix,
            build_initials_stream=build_stream_fn,
            align_acronym_to_initials=fake_align,
            kept_token_indices=lambda tokens, **kw: [0, 1, 2],
            phrase_from_indices=lambda tokens, idxs: " ".join(tokens[i] for i in idxs),
            tighten_definition_span=lambda s: s,
            normalize_definition=lambda s: s,
            is_mixed_case_acronym=lambda acr: True,
            first_alnum_char_upper=lambda tok: tok[0].upper() if tok else None,
        )

        out = find_inline_longform_after_acr("Portable Document Format", cfg, acr="pDf", require_initials_match=True)
        assert len(out) == 1

        assert seen["scan"] == "ltr"
        assert seen["expand_allcaps_tokens"] is True  # driven by is_mixed_case_acronym(acr)
        assert seen["split_compounds"] is False
        assert seen["treat_acronym_tokens_as_multi_letter"] is False

    def test_kept_token_indices_receives_bridges_and_numeric_flag(self, _patch, dummy_cfg, hit_cfg, build_stream_seen):
        cfg = dummy_cfg(max_phrase_chars=200)
        cfg.bridges = {"of"}

        state = {"n": 0}

        def fake_inline_clause_tail(_s):
            state["n"] += 1
            return (["ok"], 0) if state["n"] == 1 else ("ok", 0)

        def fake_strip_inline_cue_prefix(s, cfg):
            return "Ministry of Magic", 0

        fake_build_stream_fn, _ = build_stream_seen

        def fake_align(acr, stream, tokens, **kwargs):
            return hit_cfg(tok_left=0, tok_right=2, hit_tokens={0, 2})

        seen = {}

        def fake_kept_token_indices(tokens, **kwargs):
            seen.update(kwargs)
            return [0, 1, 2]

        _patch(
            find_inline_longform_after_acr,
            inline_clause_tail=fake_inline_clause_tail,
            collapse_ws=lambda s: " ".join(str(s).split()),
            strip_inline_cue_prefix=fake_strip_inline_cue_prefix,
            build_initials_stream=fake_build_stream_fn,
            align_acronym_to_initials=fake_align,
            kept_token_indices=fake_kept_token_indices,
            phrase_from_indices=lambda tokens, idxs: " ".join(tokens[i] for i in idxs),
            tighten_definition_span=lambda s: s,
            normalize_definition=lambda s: s,
            is_mixed_case_acronym=lambda acr: False,
            first_alnum_char_upper=lambda tok: tok[0].upper() if tok else None,
        )

        out = find_inline_longform_after_acr("Ministry of Magic", cfg, acr="MM", require_initials_match=True)
        assert len(out) == 1
        assert seen["bridges"] == {"of"}
        assert seen["include_numeric_leading"] is True

    def test_raw_window_gate_blocks(self, _patch, dummy_cfg, hit_cfg, build_stream_seen):
        cfg = dummy_cfg(max_phrase_chars=10)

        state = {"n": 0}

        def fake_inline_clause_tail(_s):
            state["n"] += 1
            return (["ok"], 0) if state["n"] == 1 else ("ok", 0)

        def fake_strip_inline_cue_prefix(s, cfg):
            # make sure ds..de slice contains the marker
            return "RAWTOOLONG token2 token3", 0

        def fake_align(acr, stream, tokens, **kwargs):
            return hit_cfg(tok_left=0, tok_right=2, hit_tokens={0, 1, 2})

        def fake_kept_token_indices(tokens, **kwargs):
            return [0, 1, 2]

        def fake_collapse_ws(s):
            txt = str(s)
            if "RAWTOOLONG" in txt:
                return "X" * 999
            return " ".join(txt.split())

        fake_build_stream_fn, _ = build_stream_seen

        _patch(
            find_inline_longform_after_acr,
            inline_clause_tail=fake_inline_clause_tail,
            collapse_ws=fake_collapse_ws,
            strip_inline_cue_prefix=fake_strip_inline_cue_prefix,
            build_initials_stream=fake_build_stream_fn,
            align_acronym_to_initials=fake_align,
            kept_token_indices=fake_kept_token_indices,
            phrase_from_indices=lambda tokens, idxs: " ".join(tokens[i] for i in idxs),
            tighten_definition_span=lambda s: s,
            normalize_definition=lambda s: s,
            is_mixed_case_acronym=lambda acr: False,
            first_alnum_char_upper=lambda tok: tok[0].upper() if tok else None,
        )

        assert (
            find_inline_longform_after_acr("RAWTOOLONG token2 token3", cfg, acr="RTT", require_initials_match=True)
            == []
        )

    def test_disp_gate_blocks_when_too_long(self, _patch, dummy_cfg, hit_cfg):
        cfg = dummy_cfg(max_phrase_chars=10)

        state = {"n": 0}

        def fake_inline_clause_tail(_s):
            state["n"] += 1
            return (["ok"], 0) if state["n"] == 1 else ("ok", 0)

        def fake_strip_inline_cue_prefix(s, cfg):
            return "Portable Document Format", 0

        def fake_build_stream(tokens, **kwargs):
            return "STREAM"

        def fake_align(acr, stream, tokens, **kwargs):
            return hit_cfg(tok_left=0, tok_right=2, hit_tokens={0, 1, 2})

        _patch(
            find_inline_longform_after_acr,
            inline_clause_tail=fake_inline_clause_tail,
            collapse_ws=lambda s: " ".join(str(s).split()),
            strip_inline_cue_prefix=fake_strip_inline_cue_prefix,
            build_initials_stream=fake_build_stream,
            align_acronym_to_initials=fake_align,
            kept_token_indices=lambda tokens, **kw: [0, 1, 2],
            phrase_from_indices=lambda tokens, idxs: " ".join(tokens[i] for i in idxs),
            tighten_definition_span=lambda s: s,
            normalize_definition=lambda s: "X" * 999,  # too long
            is_mixed_case_acronym=lambda acr: False,
            first_alnum_char_upper=lambda tok: tok[0].upper() if tok else None,
        )

        assert (
            find_inline_longform_after_acr("Portable Document Format", cfg, acr="PDF", require_initials_match=True)
            == []
        )


class TestFindInlineLongformAfterAcrUnitSearchCapAndOffsets:
    def test_respects_max_chars_search_cap(self, _patch, dummy_cfg, hit_cfg):
        cfg = dummy_cfg(max_phrase_chars=50)

        seen = {}
        state = {"n": 0}

        def fake_inline_clause_tail(s):
            state["n"] += 1
            if state["n"] == 1:
                # first call is on full snippet; must be SHORT to pass the global gate
                return "ok", 2
            # second call is on sliced `s`
            seen["s"] = s
            return s, len(s)

        def fake_strip_inline_cue_prefix(s, cfg):
            return s, 0

        def fake_align(acr, stream, tokens, **kwargs):
            return hit_cfg(tok_left=0, tok_right=min(0, len(tokens) - 1), hit_tokens={0})

        _patch(
            find_inline_longform_after_acr,
            inline_clause_tail=fake_inline_clause_tail,
            collapse_ws=lambda s: s,  # identity is fine here
            strip_inline_cue_prefix=fake_strip_inline_cue_prefix,
            build_initials_stream=lambda tokens, **kw: "STREAM",
            align_acronym_to_initials=fake_align,
            kept_token_indices=lambda tokens, **kw: [0],
            phrase_from_indices=lambda tokens, idxs: tokens[idxs[0]],
            tighten_definition_span=lambda s: s,
            normalize_definition=lambda s: s,
            is_mixed_case_acronym=lambda acr: False,
            first_alnum_char_upper=lambda tok: tok[0].upper() if tok else None,
        )

        snippet = "A" * 1000
        out = find_inline_longform_after_acr(snippet, cfg, acr="A", max_chars=10, require_initials_match=True)

        assert len(out) == 1
        assert seen["s"] == "A" * 10

    def test_offset_from_cue_is_applied_to_spans(self, _patch, dummy_cfg, hit_cfg):
        cfg = dummy_cfg(max_phrase_chars=200)

        state = {"n": 0}

        def fake_inline_clause_tail(s):
            state["n"] += 1
            return (["ok"], 0) if state["n"] == 1 else (s, 0)

        # Simulate cue stripping: tail2 starts later in the string.
        # We return off=5, and ensure ds..de point into original snippet.
        def fake_strip_inline_cue_prefix(s, cfg):
            return "Portable Document Format", 5

        def fake_align(acr, stream, tokens, **kwargs):
            return hit_cfg(tok_left=0, tok_right=2, hit_tokens={0, 1, 2})

        _patch(
            find_inline_longform_after_acr,
            inline_clause_tail=fake_inline_clause_tail,
            collapse_ws=lambda s: " ".join(str(s).split()),
            strip_inline_cue_prefix=fake_strip_inline_cue_prefix,
            build_initials_stream=lambda tokens, **kw: "STREAM",
            align_acronym_to_initials=fake_align,
            kept_token_indices=lambda tokens, **kw: [0, 1, 2],
            phrase_from_indices=lambda tokens, idxs: " ".join(tokens[i] for i in idxs),
            tighten_definition_span=lambda s: s,
            normalize_definition=lambda s: s,
            is_mixed_case_acronym=lambda acr: False,
            first_alnum_char_upper=lambda tok: tok[0].upper() if tok else None,
        )

        snippet = "CUE: Portable Document Format"
        out = find_inline_longform_after_acr(snippet, cfg, acr="PDF", require_initials_match=True)
        assert len(out) == 1
        m = out[0]
        assert snippet[m.def_start : m.def_end] == "Portable Document Format"


class TestFindInlineLongformAfterAcrUnitWindowingAndFailureModes:
    def test_span_hugs_first_and_last_kept_indices(self, _patch, dummy_cfg, hit_cfg):
        cfg = dummy_cfg(max_phrase_chars=200)

        state = {"n": 0}

        def fake_inline_clause_tail(s):
            state["n"] += 1
            return (["ok"], 0) if state["n"] == 1 else (s, 0)

        def fake_strip_inline_cue_prefix(s, cfg):
            return "Alpha Beta Gamma Delta", 0

        def fake_align(acr, stream, tokens, **kwargs):
            return hit_cfg(tok_left=0, tok_right=3, hit_tokens={0, 3})

        # Keep non-contiguous tokens: Alpha + Gamma (idx 0 and 2)
        def fake_kept_token_indices(tokens, **kwargs):
            return [0, 2]

        _patch(
            find_inline_longform_after_acr,
            inline_clause_tail=fake_inline_clause_tail,
            collapse_ws=lambda s: " ".join(str(s).split()),
            strip_inline_cue_prefix=fake_strip_inline_cue_prefix,
            build_initials_stream=lambda tokens, **kw: "STREAM",
            align_acronym_to_initials=fake_align,
            kept_token_indices=fake_kept_token_indices,
            phrase_from_indices=lambda tokens, idxs: " ".join(tokens[i] for i in idxs),
            tighten_definition_span=lambda s: s,
            normalize_definition=lambda s: s,
            is_mixed_case_acronym=lambda acr: False,
            first_alnum_char_upper=lambda tok: tok[0].upper() if tok else None,
        )

        snippet = "Alpha Beta Gamma Delta"
        out = find_inline_longform_after_acr(snippet, cfg, acr="AG", require_initials_match=True)
        assert len(out) == 1
        m = out[0]
        # Span should include from start of Alpha to end of Gamma (because kept_idx[-1] == 2)
        assert snippet[m.def_start : m.def_end] == "Alpha Beta Gamma"
        assert m.definition == "Alpha Gamma"

    def test_returns_empty_when_no_tokens_after_cue(self, _patch, dummy_cfg):
        cfg = dummy_cfg(max_phrase_chars=200)

        state = {"n": 0}

        def fake_inline_clause_tail(s):
            state["n"] += 1
            return (["ok"], 0) if state["n"] == 1 else (s, 0)

        def fake_strip_inline_cue_prefix(s, cfg):
            return "", 0

        _patch(
            find_inline_longform_after_acr,
            inline_clause_tail=fake_inline_clause_tail,
            collapse_ws=lambda s: s,
            strip_inline_cue_prefix=fake_strip_inline_cue_prefix,
        )

        assert find_inline_longform_after_acr("CUE:", cfg, acr="X", require_initials_match=True) == []

    def test_returns_empty_when_kept_indices_empty(self, _patch, dummy_cfg, hit_cfg):
        cfg = dummy_cfg(max_phrase_chars=200)

        state = {"n": 0}

        def fake_inline_clause_tail(s):
            state["n"] += 1
            return (["ok"], 0) if state["n"] == 1 else (s, 0)

        def fake_strip_inline_cue_prefix(s, cfg):
            return "Portable Document Format", 0

        def fake_align(acr, stream, tokens, **kwargs):
            return hit_cfg(tok_left=0, tok_right=2, hit_tokens={0, 1, 2})

        _patch(
            find_inline_longform_after_acr,
            inline_clause_tail=fake_inline_clause_tail,
            collapse_ws=lambda s: " ".join(str(s).split()),
            strip_inline_cue_prefix=fake_strip_inline_cue_prefix,
            build_initials_stream=lambda tokens, **kw: "STREAM",
            align_acronym_to_initials=fake_align,
            kept_token_indices=lambda tokens, **kw: [],
        )

        # kept_idx[0] would crash; this test asserts the function should return []
        # If it currently crashes, that’s a bug worth fixing (guard kept_idx).
        assert (
            find_inline_longform_after_acr("Portable Document Format", cfg, acr="PDF", require_initials_match=True)
            == []
        )

    def test_returns_empty_when_phrase_from_indices_empty(self, _patch, dummy_cfg, hit_cfg):
        cfg = dummy_cfg(max_phrase_chars=200)

        state = {"n": 0}

        def fake_inline_clause_tail(s):
            state["n"] += 1
            return (["ok"], 0) if state["n"] == 1 else (s, 0)

        def fake_strip_inline_cue_prefix(s, cfg):
            return "Portable Document Format", 0

        def fake_align(acr, stream, tokens, **kwargs):
            return hit_cfg(tok_left=0, tok_right=2, hit_tokens={0, 1, 2})

        _patch(
            find_inline_longform_after_acr,
            inline_clause_tail=fake_inline_clause_tail,
            collapse_ws=lambda s: " ".join(str(s).split()),
            strip_inline_cue_prefix=fake_strip_inline_cue_prefix,
            build_initials_stream=lambda tokens, **kw: "STREAM",
            align_acronym_to_initials=fake_align,
            kept_token_indices=lambda tokens, **kw: [0, 1, 2],
            phrase_from_indices=lambda tokens, idxs: "",
            tighten_definition_span=lambda s: s,
            normalize_definition=lambda s: s,
            is_mixed_case_acronym=lambda acr: False,
            first_alnum_char_upper=lambda tok: tok[0].upper() if tok else None,
        )

        assert (
            find_inline_longform_after_acr("Portable Document Format", cfg, acr="PDF", require_initials_match=True)
            == []
        )

    def test_returns_empty_when_tighten_or_normalize_returns_falsy(self, _patch, dummy_cfg, hit_cfg):
        cfg = dummy_cfg(max_phrase_chars=200)

        state = {"n": 0}

        def fake_inline_clause_tail(s):
            state["n"] += 1
            return (["ok"], 0) if state["n"] == 1 else (s, 0)

        def fake_strip_inline_cue_prefix(s, cfg):
            return "Portable Document Format", 0

        def fake_align(acr, stream, tokens, **kwargs):
            return hit_cfg(tok_left=0, tok_right=2, hit_tokens={0, 1, 2})

        _patch(
            find_inline_longform_after_acr,
            inline_clause_tail=fake_inline_clause_tail,
            collapse_ws=lambda s: " ".join(str(s).split()),
            strip_inline_cue_prefix=fake_strip_inline_cue_prefix,
            build_initials_stream=lambda tokens, **kw: "STREAM",
            align_acronym_to_initials=fake_align,
            kept_token_indices=lambda tokens, **kw: [0, 1, 2],
            phrase_from_indices=lambda tokens, idxs: "Portable Document Format",
            tighten_definition_span=lambda s: "",
            normalize_definition=lambda s: "",
            is_mixed_case_acronym=lambda acr: False,
            first_alnum_char_upper=lambda tok: tok[0].upper() if tok else None,
        )

        assert (
            find_inline_longform_after_acr("Portable Document Format", cfg, acr="PDF", require_initials_match=True)
            == []
        )


class TestFindInlineLongformAfterAcrUnitConfigKeys:
    def test_uses_cfg_stop_attribute_not_stopwords(self, _patch, dummy_cfg, hit_cfg):
        cfg = dummy_cfg(max_phrase_chars=200)
        cfg.stop = {"the"}  # the function reads cfg.stop, not cfg.stopwords

        state = {"n": 0}

        def fake_inline_clause_tail(s):
            state["n"] += 1
            return (["ok"], 0) if state["n"] == 1 else (s, 0)

        def fake_strip_inline_cue_prefix(s, cfg):
            return "Portable Document Format", 0

        seen = {}

        def fake_build_stream(tokens, **kwargs):
            seen.update(kwargs)
            return "STREAM"

        def fake_align(acr, stream, tokens, **kwargs):
            return hit_cfg(tok_left=0, tok_right=2, hit_tokens={0, 1, 2})

        _patch(
            find_inline_longform_after_acr,
            inline_clause_tail=fake_inline_clause_tail,
            collapse_ws=lambda s: " ".join(str(s).split()),
            strip_inline_cue_prefix=fake_strip_inline_cue_prefix,
            build_initials_stream=fake_build_stream,
            align_acronym_to_initials=fake_align,
            kept_token_indices=lambda tokens, **kw: [0, 1, 2],
            phrase_from_indices=lambda tokens, idxs: " ".join(tokens[i] for i in idxs),
            tighten_definition_span=lambda s: s,
            normalize_definition=lambda s: s,
            is_mixed_case_acronym=lambda acr: False,
            first_alnum_char_upper=lambda tok: tok[0].upper() if tok else None,
        )

        out = find_inline_longform_after_acr("Portable Document Format", cfg, acr="PDF", require_initials_match=True)
        assert len(out) == 1
        assert seen["stopwords"] == {"the"}
