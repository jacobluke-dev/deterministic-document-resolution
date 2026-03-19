from plainera_unacronym.nlp.extraction.acronyms.matchers.defs import find_parenthetical_longform_before_acr


class TestFindParentheticalLongformBeforeAcrIntegration:
    def test_basic_forward_phrase_then_acr(self, dummy_cfg):
        cfg = dummy_cfg()
        snippet = "Portable Document Format (PDF)"
        out = find_parenthetical_longform_before_acr(snippet, "PDF", cfg)
        assert len(out) == 1
        assert out[0].definition == "Portable Document Format"

    def test_trailing_punctuation_is_normalized(self, dummy_cfg):
        cfg = dummy_cfg()
        snippet = "Graphics   Processing   Unit...   (GPU)"
        out = find_parenthetical_longform_before_acr(snippet, "GPU", cfg)
        assert len(out) == 1
        assert out[0].definition == "Graphics Processing Unit"

    def test_titlecase_tail_preference_is_respected(self, dummy_cfg):
        cfg = dummy_cfg()
        # If the tighten_definition_span favors the last TitleCase/UPPER chunk,
        # ensure we still get the meaningful tail.
        snippet = "See also the HyperText Transfer Protocol (HTTP)"
        out = find_parenthetical_longform_before_acr(snippet, "HTTP", cfg)
        assert len(out) == 1
        assert out[0].definition == "HyperText Transfer Protocol"

    def test_boundary_and_whitespace_variants(self, dummy_cfg):
        cfg = dummy_cfg()
        snippet = "Read Only Memory   ( ROM )   "
        out = find_parenthetical_longform_before_acr(snippet, "ROM", cfg)
        assert len(out) == 1
        assert out[0].definition == "Read Only Memory"

    def test_respects_max_chars_integration(self, dummy_cfg):
        cfg = dummy_cfg(max_phrase_chars=10)
        snippet = "Hypertext Transfer Protocol (HTTP)"
        # Def > 10 chars → no match
        assert find_parenthetical_longform_before_acr(snippet, "HTTP", cfg) == []


class TestFindParentheticalLongformBeforeAcrUnitAlignmentAndFallback:
    def test_align_fallback_called_when_strict_fails(self, _patch, dummy_cfg, hit_cfg, build_stream_seen):
        build_stream_fn, _ = build_stream_seen

        calls = {"allow_upper_on_stop": []}

        def fake_align(acr, stream, tokens, **kwargs):
            calls["allow_upper_on_stop"].append(kwargs.get("allow_upper_on_stop"))
            if kwargs.get("allow_upper_on_stop") is False:
                return None
            return hit_cfg(tok_left=0, hit_tokens={0, 1, 2})

        def fake_build_kept_phrase(_tokens, **kw):
            return "Portable Document Format"

        _patch(
            find_parenthetical_longform_before_acr,
            has_letter=lambda s: True,
            build_initials_stream=build_stream_fn,
            align_acronym_to_initials=fake_align,
            expand_numeric_leading_window=lambda tokens, i, j: (i, j),
            consume_left_numeric_designator=lambda acr, tokens, tok_left: tok_left,
            build_kept_phrase=fake_build_kept_phrase,
            normalize_definition=lambda s: "Portable Document Format",
            collapse_ws=lambda s: " ".join(s.split()),
        )

        cfg = dummy_cfg()
        snippet = "Portable Document Format (PDF)"
        out = find_parenthetical_longform_before_acr(snippet, "PDF", cfg)
        assert len(out) == 1
        assert calls["allow_upper_on_stop"] == [False, True]

    def test_build_initials_stream_called_with_expected_flags(self, _patch, dummy_cfg, hit_cfg, build_stream_seen):
        seen = {}

        build_stream_seen, seen = build_stream_seen

        def fake_align(acr, stream, tokens, **kwargs):
            return hit_cfg(tok_left=0, hit_tokens={0, 1, 2})

        def fake_build_kept_phrase(_tokens, **kw):
            return "Portable Document Format"

        _patch(
            find_parenthetical_longform_before_acr,
            has_letter=lambda s: True,
            build_initials_stream=build_stream_seen,
            align_acronym_to_initials=fake_align,
            expand_numeric_leading_window=lambda tokens, i, j: (i, j),
            consume_left_numeric_designator=lambda acr, tokens, tok_left: tok_left,
            build_kept_phrase=fake_build_kept_phrase,
            normalize_definition=lambda s: "Portable Document Format",
            collapse_ws=lambda s: " ".join(s.split()),
            is_mixed_case_acronym=lambda acr: False,
        )

        cfg = dummy_cfg()
        snippet = "Portable Document Format (PDF)"
        out = find_parenthetical_longform_before_acr(snippet, "PDF", cfg)
        assert len(out) == 1

        assert seen["scan"] == "rtl"
        assert seen["expand_allcaps_tokens"] is False
        assert seen["split_compounds"] is False
        assert seen["treat_acronym_tokens_as_multi_letter"] is False

    def test_mixed_case_acronym_controls_allow_lower_on_non_stop(self, _patch, dummy_cfg, hit_cfg, build_stream_seen):
        calls = {"allow_lower_on_non_stop": []}
        build_stream_fn, _ = build_stream_seen

        def fake_align(acr, stream, tokens, **kwargs):
            calls["allow_lower_on_non_stop"].append(kwargs.get("allow_lower_on_non_stop"))
            return hit_cfg(tok_left=0, hit_tokens={0, 1})

        def fake_build_kept_phrase(_tokens, **kw):
            return "electric Grid"

        _patch(
            find_parenthetical_longform_before_acr,
            has_letter=lambda s: True,
            build_initials_stream=build_stream_fn,
            align_acronym_to_initials=fake_align,
            expand_numeric_leading_window=lambda tokens, i, j: (i, j),
            consume_left_numeric_designator=lambda acr, tokens, tok_left: tok_left,
            build_kept_phrase=fake_build_kept_phrase,
            normalize_definition=lambda s: s.strip(),
            collapse_ws=lambda s: " ".join(s.split()),
            is_mixed_case_acronym=lambda acr: True,
        )

        cfg = dummy_cfg()
        out = find_parenthetical_longform_before_acr("electric Grid (eG)", "eG", cfg)
        assert len(out) == 1
        assert all(v is True for v in calls["allow_lower_on_non_stop"])


class TestFindParentheticalLongformBeforeAcrUnitNumericDesignatorsAndWindowing:
    def test_consume_left_numeric_designator_called_twice_when_acr_starts_with_digit(
        self, _patch, dummy_cfg, hit_cfg, build_stream_seen
    ):
        build_stream_fn, _ = build_stream_seen

        calls = {"consume": 0}

        def fake_align(acr, stream, tokens, **kwargs):
            return hit_cfg(tok_left=1, hit_tokens={1, 2})

        def fake_consume(*, acr, tokens, tok_left):
            calls["consume"] += 1
            return tok_left - 1

        def fake_build_kept_phrase(_tokens, **kw):
            return "3M Portable format"

        _patch(
            find_parenthetical_longform_before_acr,
            has_letter=lambda s: True,
            build_initials_stream=build_stream_fn,
            align_acronym_to_initials=fake_align,
            consume_left_numeric_designator=fake_consume,
            expand_numeric_leading_window=lambda tokens, i, j: (i, j),
            build_kept_phrase=fake_build_kept_phrase,
            normalize_definition=lambda s: s,
            collapse_ws=lambda s: " ".join(s.split()),
        )

        cfg = dummy_cfg()
        snippet = "3M Portable format (3M)"
        out = find_parenthetical_longform_before_acr(snippet, "3M", cfg)
        assert len(out) == 1
        assert out[0].definition == "3M Portable format"
        assert calls["consume"] == 2

    def test_expand_numeric_leading_window_parameters_used(self, _patch, dummy_cfg, hit_cfg, build_stream_seen):
        seen = {}
        build_stream_fn, seen = build_stream_seen

        def fake_align(acr, stream, tokens, **kwargs):
            return hit_cfg(tok_left=1, hit_tokens={1, 2})

        def fake_expand(tokens, i, j):
            seen["before"] = (i, j)
            return (0, j)

        def fake_build_kept_phrase(_tokens, **kw):
            seen["tok_left"] = kw["tok_left"]
            seen["tok_right"] = kw["tok_right"]
            return "3M Portable format"

        _patch(
            find_parenthetical_longform_before_acr,
            has_letter=lambda s: True,
            build_initials_stream=build_stream_fn,
            align_acronym_to_initials=fake_align,
            consume_left_numeric_designator=lambda **kw: kw["tok_left"],
            expand_numeric_leading_window=fake_expand,
            build_kept_phrase=fake_build_kept_phrase,
            normalize_definition=lambda s: s,
            collapse_ws=lambda s: " ".join(s.split()),
        )

        cfg = dummy_cfg()
        snippet = "3M Portable format (PF)"
        out = find_parenthetical_longform_before_acr(snippet, "PF", cfg)
        assert len(out) == 1
        assert seen["before"] == (1, 2)
        assert seen["tok_left"] == 0
        assert seen["tok_right"] == 2

    def test_build_kept_phrase_receives_bridges_and_numeric_flag(self, _patch, dummy_cfg, hit_cfg, build_stream_seen):
        seen = {}
        build_stream_fn, seen = build_stream_seen

        def fake_align(acr, stream, tokens, **kwargs):
            return hit_cfg(tok_left=0, hit_tokens={0, 2})

        def fake_build_kept_phrase(_tokens, **kw):
            seen["bridges"] = kw["bridges"]
            seen["include_numeric_leading"] = kw["include_numeric_leading"]
            return "Ministry of Magic"

        _patch(
            find_parenthetical_longform_before_acr,
            has_letter=lambda s: True,
            build_initials_stream=build_stream_fn,
            align_acronym_to_initials=fake_align,
            consume_left_numeric_designator=lambda **kw: kw["tok_left"],
            expand_numeric_leading_window=lambda tokens, i, j: (i, j),
            build_kept_phrase=fake_build_kept_phrase,
            normalize_definition=lambda s: s,
            collapse_ws=lambda s: " ".join(s.split()),
        )

        cfg = dummy_cfg()
        cfg.bridges = {"of"}
        snippet = "Ministry of Magic (MM)"
        out = find_parenthetical_longform_before_acr(snippet, "MM", cfg)
        assert len(out) == 1
        assert seen["bridges"] == {"of"}
        assert seen["include_numeric_leading"] is True


class TestFindParentheticalLongformBeforeAcrUnitNormalizeFailures:
    def test_returns_empty_if_normalize_definition_falsy(self, _patch, dummy_cfg, hit_cfg, build_stream_seen):
        build_stream_fn, _ = build_stream_seen

        def fake_align(acr, stream, tokens, **kwargs):
            return hit_cfg(tok_left=0, hit_tokens={0, 1, 2})

        def fake_build_kept_phrase(_tokens, **kw):
            return "Portable Document Format"

        _patch(
            find_parenthetical_longform_before_acr,
            has_letter=lambda s: True,
            build_initials_stream=build_stream_fn,
            align_acronym_to_initials=fake_align,
            consume_left_numeric_designator=lambda **kw: kw["tok_left"],
            expand_numeric_leading_window=lambda tokens, i, j: (i, j),
            build_kept_phrase=fake_build_kept_phrase,
            normalize_definition=lambda s: "",  # force failure
            collapse_ws=lambda s: " ".join(s.split()),
        )

        cfg = dummy_cfg()
        snippet = "Portable Document Format (PDF)"
        assert find_parenthetical_longform_before_acr(snippet, "PDF", cfg) == []
