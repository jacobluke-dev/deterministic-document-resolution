from plainera_unacronym.nlp.extraction.matchers.defs import find_parenthetical_longform_after_acr


class TestFindParentheticalLongformAfterAcrUnit:
    def test_no_parenthesized_match_returns_empty(self, _patch, dummy_cfg):
        # Patching anyway to prove independence; they won't be called
        _patch(
            find_parenthetical_longform_after_acr,
            has_letter=lambda s: True,
            tighten_definition_span=lambda s: s,
            normalize_definition=lambda s: s,
            _initials_match=lambda acr, phrase: True,
        )
        cfg = dummy_cfg()
        assert find_parenthetical_longform_after_acr("no parens here", cfg, acr="PDF") == []

    def test_requires_letters_gate(self, _patch, dummy_cfg):
        calls = {}

        def spy_has_letter(s):
            calls["raw"] = s
            return False  # force gate fail

        _patch(
            find_parenthetical_longform_after_acr,
            has_letter=spy_has_letter,
            tighten_definition_span=lambda s: "IGNORED",
            normalize_definition=lambda s: "IGNORED",
            _initials_match=lambda acr, phrase: True,
        )
        cfg = dummy_cfg()
        snip = "   (1234) tail"
        assert find_parenthetical_longform_after_acr(snip, cfg, acr="X") == []
        # ensure we passed the raw inner text to _has_letter
        assert calls["raw"] == "1234"

    def test_normalize_pipeline_and_span_preserved(self, _patch, dummy_cfg):
        seen = {}

        def fake_tighten(s):
            seen["tighten_in"] = s
            return " Foo   Bar... "

        def fake_normalize(s):
            seen["normalize_in"] = s
            return "Foo Bar"  # collapsed + stripped

        _patch(
            find_parenthetical_longform_after_acr,
            has_letter=lambda s: True,
            tighten_definition_span=fake_tighten,
            normalize_definition=fake_normalize,
            _initials_match=lambda acr, phrase: True,
        )
        cfg = dummy_cfg()
        raw = " noisy    RAW "
        snip = f"  ({raw}) and more"
        out = find_parenthetical_longform_after_acr(snip, cfg, acr="FB", require_initials_match=False)
        assert len(out) == 1
        m = out[0]
        # Output definition is the normalized value
        assert m.definition == "Foo Bar"

        # Indices hug the content (no inner padding)
        assert snip[m.def_start : m.def_end] == raw.strip()

        # Verify pipeline call args: we now feed the *tight* captured def
        assert seen["tighten_in"] == raw.strip()

        # And normalize is called with whatever tighten returned
        assert seen["normalize_in"] == " Foo   Bar... "

    def test_require_initials_match_guard_true_allows(self, _patch, dummy_cfg):
        seen = {}

        def fake_tighten(s):
            seen["tighten_in"] = s
            return s  # or " Foo   Bar... " if test normalization too

        def fake_normalize(s):
            seen["normalize_in"] = s
            return s.strip()

        _patch(
            find_parenthetical_longform_after_acr,
            has_letter=lambda s: True,
            tighten_definition_span=fake_tighten,
            normalize_definition=fake_normalize,
        )

        cfg = dummy_cfg()
        snip = "(Portable Document Format)"
        out = find_parenthetical_longform_after_acr(snip, cfg, acr="PDF", require_initials_match=False)
        assert len(out) == 1
        assert out[0].definition == "Portable Document Format"

        # (Optional) verify the pipeline inputs were what can be expected
        assert seen["tighten_in"] == "Portable Document Format"
        assert seen["normalize_in"] == "Portable Document Format"

    def test_require_initials_match_guard_false_blocks(self, _patch, dummy_cfg):
        _patch(
            find_parenthetical_longform_after_acr,
            has_letter=lambda s: True,
            tighten_definition_span=lambda s: s,
            normalize_definition=lambda s: s,
        )
        cfg = dummy_cfg()
        snip = "Portable Document Format"
        assert find_parenthetical_longform_after_acr(snip, cfg, acr="PDF", require_initials_match=True) == []

    def test_max_chars_respected(self, _patch, dummy_cfg):
        cfg = dummy_cfg(max_phrase_chars=3)
        assert find_parenthetical_longform_after_acr("(Portable)", cfg, acr="P") == []


class TestFindParentheticalLongformAfterAcrUnitInitialsPath:
    def test_alignment_success_uses_window_and_span(self, _patch, dummy_cfg, hit_cfg, build_stream_seen):
        # tokens: Portable Document Format -> pick [0..2]
        build_stream_fn, _ = build_stream_seen

        def fake_align(acr, stream, tokens, **kwargs):
            return hit_cfg(tok_left=0, tok_right=2, hit_tokens={0, 1, 2})

        _patch(
            find_parenthetical_longform_after_acr,
            has_letter=lambda s: True,
            build_initials_stream=build_stream_fn,
            align_acronym_to_initials=fake_align,
            expand_numeric_leading_window=lambda tokens, i, j: (i, j),
            normalize_definition=lambda s: s,
            tighten_definition_span=lambda s: s,
            strip_trailing_punct_str=lambda s: s,
            collapse_ws=lambda s: " ".join(s.split()),
        )

        cfg = dummy_cfg()
        snip = " (Portable Document Format) trailing"
        out = find_parenthetical_longform_after_acr(snip, cfg, acr="PDF", require_initials_match=True)
        assert len(out) == 1
        m = out[0]
        assert m.definition == "Portable Document Format"
        assert snip[m.def_start : m.def_end] == "Portable Document Format"

    def test_alignment_fallback_when_upper_on_stop_disallowed_then_allowed(
        self, _patch, dummy_cfg, hit_cfg, build_stream_seen
    ):
        calls = {"align": []}

        build_stream_fn, _ = build_stream_seen

        def fake_align(acr, stream, tokens, **kwargs):
            # First call: allow_upper_on_stop=False -> fail
            # Second call: allow_upper_on_stop=True -> succeed
            calls["align"].append(kwargs.get("allow_upper_on_stop"))
            if kwargs.get("allow_upper_on_stop") is False:
                return None
            return hit_cfg(tok_left=0, tok_right=2, hit_tokens={0, 2})  # intentionally omit middle token

        _patch(
            find_parenthetical_longform_after_acr,
            has_letter=lambda s: True,
            build_initials_stream=build_stream_fn,
            align_acronym_to_initials=fake_align,
            expand_numeric_leading_window=lambda tokens, i, j: (i, j),
            normalize_definition=lambda s: s,
            tighten_definition_span=lambda s: s,
            strip_trailing_punct_str=lambda s: s,
            collapse_ws=lambda s: " ".join(s.split()),
        )

        cfg = dummy_cfg()
        snip = "(Magic of Oz)"
        out = find_parenthetical_longform_after_acr(snip, cfg, acr="MO", require_initials_match=True)
        assert len(out) == 1
        # Ensure fallback path was taken (False then True)
        assert calls["align"] == [False, True]

    def test_mixed_case_acronym_sets_allow_lower_on_non_stop_true(self, _patch, dummy_cfg, hit_cfg, build_stream_seen):
        calls = {}

        build_stream_fn, _ = build_stream_seen

        def fake_align(acr, stream, tokens, **kwargs):
            calls["allow_lower_on_non_stop"] = kwargs.get("allow_lower_on_non_stop")
            return hit_cfg(tok_left=0, tok_right=1, hit_tokens={0, 1})

        _patch(
            find_parenthetical_longform_after_acr,
            has_letter=lambda s: True,
            is_mixed_case_acronym=lambda acr: True,
            build_initials_stream=build_stream_fn,
            align_acronym_to_initials=fake_align,
            expand_numeric_leading_window=lambda tokens, i, j: (i, j),
            normalize_definition=lambda s: s,
            tighten_definition_span=lambda s: s,
            strip_trailing_punct_str=lambda s: s,
            collapse_ws=lambda s: " ".join(s.split()),
        )

        cfg = dummy_cfg()
        out = find_parenthetical_longform_after_acr("(electric Grid)", cfg, acr="eG", require_initials_match=True)
        assert len(out) == 1
        assert calls["allow_lower_on_non_stop"] is True


class TestFindParentheticalLongformAfterAcrUnitKeptTokens:
    def test_bridges_are_kept_even_if_not_hit_token(self, _patch, dummy_cfg, hit_cfg, build_stream_seen):
        # Hit tokens only include Ministry + Magic, but "of" should remain if cfg.bridges includes it
        build_stream_fn, _ = build_stream_seen

        def fake_align(acr, stream, tokens, **kwargs):
            return hit_cfg(tok_left=0, tok_right=2, hit_tokens={0, 2})

        _patch(
            find_parenthetical_longform_after_acr,
            has_letter=lambda s: True,
            build_initials_stream=build_stream_fn,
            align_acronym_to_initials=fake_align,
            expand_numeric_leading_window=lambda tokens, i, j: (i, j),
            normalize_definition=lambda s: s,
            tighten_definition_span=lambda s: s,
            strip_trailing_punct_str=lambda s: s,
            collapse_ws=lambda s: " ".join(s.split()),
        )

        cfg = dummy_cfg()
        cfg.bridges = {"of"}  # override bridges explicitly
        out = find_parenthetical_longform_after_acr("(Ministry of Magic)", cfg, acr="MM", require_initials_match=True)
        assert len(out) == 1
        assert out[0].definition == "Ministry of Magic"

    def test_expand_numeric_leading_window_can_extend_left_span(self, _patch, dummy_cfg, hit_cfg, build_stream_seen):
        build_stream_fn, _ = build_stream_seen

        def fake_align(acr, stream, tokens, **kwargs):
            # pretend acronym hits "Portable format" only
            return hit_cfg(tok_left=1, tok_right=2, hit_tokens={1, 2})

        def fake_expand(tokens, i, j):
            # expand to include numeric-leading token
            return (0, j)

        _patch(
            find_parenthetical_longform_after_acr,
            has_letter=lambda s: True,
            build_initials_stream=build_stream_fn,
            align_acronym_to_initials=fake_align,
            expand_numeric_leading_window=fake_expand,
            normalize_definition=lambda s: s,
            tighten_definition_span=lambda s: s,
            strip_trailing_punct_str=lambda s: s,
            collapse_ws=lambda s: " ".join(s.split()),
            # treat 3M as numeric-ish so it stays even if not hit token
            first_alnum_char_upper=lambda tok: "3" if tok.startswith("3") else tok[0].upper(),
        )

        cfg = dummy_cfg()
        snip = "(3M Portable format) trailing"
        out = find_parenthetical_longform_after_acr(snip, cfg, acr="PF", require_initials_match=True)
        assert len(out) == 1
        m = out[0]
        assert m.definition == "3M Portable format"
        assert snip[m.def_start : m.def_end] == "3M Portable format"


class TestFindParentheticalLongformAfterAcrUnitSpans:
    def test_repeated_tokens_span_points_to_correct_occurrence(self, _patch, dummy_cfg, hit_cfg, build_stream_seen):
        build_stream_fn, _ = build_stream_seen

        def fake_align(acr, stream, tokens, **kwargs):
            # pick the last two "A" tokens => indices 1..2
            return hit_cfg(tok_left=1, tok_right=2, hit_tokens={1, 2})

        _patch(
            find_parenthetical_longform_after_acr,
            has_letter=lambda s: True,
            build_initials_stream=build_stream_fn,
            align_acronym_to_initials=fake_align,
            expand_numeric_leading_window=lambda tokens, i, j: (i, j),
            normalize_definition=lambda s: s,
            tighten_definition_span=lambda s: s,
            strip_trailing_punct_str=lambda s: s,
            collapse_ws=lambda s: " ".join(s.split()),
            first_alnum_char_upper=lambda tok: tok[0].upper(),
        )

        cfg = dummy_cfg()
        snip = "(A A A) tail"
        out = find_parenthetical_longform_after_acr(snip, cfg, acr="AA", require_initials_match=True)
        assert len(out) == 1
        m = out[0]
        assert snip[m.def_start : m.def_end] == "A A"
        assert m.definition == "A A"


class TestFindParentheticalLongformAfterAcrUnitMissingAcronymBehaviour:
    def test_require_initials_match_true_but_acr_none_currently_accepts_all(self, dummy_cfg):
        cfg = dummy_cfg()
        out = find_parenthetical_longform_after_acr(
            "(Portable Document Format)", cfg, acr=None, require_initials_match=True
        )
        assert len(out) == 1
        assert out[0].definition == "Portable Document Format"

    def test_require_initials_match_true_but_acr_empty_string_currently_accepts_all(self, dummy_cfg):
        cfg = dummy_cfg()
        out = find_parenthetical_longform_after_acr(
            "(Portable Document Format)", cfg, acr="", require_initials_match=True
        )
        assert len(out) == 1
        assert out[0].definition == "Portable Document Format"


class TestFindLongformAfterAcrIntegration:
    def test_no_parenthesized_match_returns_empty(self, dummy_cfg):
        cfg = dummy_cfg()
        snippet = "  not a parenthetical here"
        assert find_parenthetical_longform_after_acr(snippet, cfg, acr="PDF") == []

    def test_requires_letters(self, dummy_cfg):
        # "(1234)" contains no letters, should be rejected
        cfg = dummy_cfg()
        snippet = "   (1234)   trailing"
        assert find_parenthetical_longform_after_acr(snippet, cfg, acr="PDF") == []

    def test_respects_max_chars(self, dummy_cfg):
        cfg = dummy_cfg(max_phrase_chars=5)
        # "Portable" exceeds max=5, so regex won't match at all
        snippet = " (Portable) "
        assert find_parenthetical_longform_after_acr(snippet, cfg, acr="P") == []

        # Fits within the limit
        cfg2 = dummy_cfg(max_phrase_chars=12)
        snippet2 = " (Portable) "
        out = find_parenthetical_longform_after_acr(snippet2, cfg2, acr="P")
        assert len(out) == 1
        assert out[0].definition == "Portable"

    def test_normalization_pipeline_applied(self, dummy_cfg):
        cfg = dummy_cfg()
        # Extra whitespace + trailing punctuation → normalized by tighten_definition_span + normalize_definition
        snippet = "   (  Portable   Document   Format...   )  "
        out = find_parenthetical_longform_after_acr(snippet, cfg, acr="PDF")
        assert len(out) == 1
        assert out[0].definition == "Portable Document Format"

    def test_require_initials_match_guard_allows_good_match(self, dummy_cfg):
        cfg = dummy_cfg()
        snippet = " (Graphics Processing Unit) "
        out = find_parenthetical_longform_after_acr(snippet, cfg, acr="GPU", require_initials_match=True)
        assert len(out) == 1
        assert out[0].definition == "Graphics Processing Unit"

    def test_require_initials_match_guard_blocks_bad_match(self, dummy_cfg):
        cfg = dummy_cfg()
        snippet = " (Portable Document Format) "
        # Wrong order: 'PFD' does not fit initials 'PDF'
        out = find_parenthetical_longform_after_acr(snippet, cfg, acr="PFD", require_initials_match=True)
        assert out == []

    def test_disable_require_initials_match_guard(self, dummy_cfg):
        cfg = dummy_cfg()
        snippet = " (Portable Document Format) "
        out = find_parenthetical_longform_after_acr(snippet, cfg, acr="PFD", require_initials_match=False)
        assert len(out) == 1
        assert out[0].definition == "Portable Document Format"

    def test_def_span_indices_are_correct(self, dummy_cfg):
        cfg = dummy_cfg()
        raw_def = "Portable Document Format"
        snippet = f"   ({raw_def}) and more"
        out = find_parenthetical_longform_after_acr(snippet, cfg, acr="PDF")
        assert len(out) == 1
        m = out[0]
        # Ensure the span points exactly to the definition characters within snippet
        assert snippet[m.def_start : m.def_end] == raw_def

    def test_forward_form_pdf(self, dummy_cfg):
        cfg = dummy_cfg()
        # Caller slices snippet to start at acr_end; we simulate by starting at '('
        snippet = "(Portable Document Format) please proceed"
        out = find_parenthetical_longform_after_acr(snippet, cfg, acr="PDF", require_initials_match=True)
        assert len(out) == 1
        item = out[0]
        assert item.definition == "Portable Document Format"
        assert snippet[item.def_start : item.def_end] == "Portable Document Format"

    def test_whitespace_and_punct_cleaned(self, dummy_cfg):
        cfg = dummy_cfg()
        snippet = "   (  Graphics    Processing  Unit... ) more"
        out = find_parenthetical_longform_after_acr(snippet, cfg, acr="GPU", require_initials_match=True)
        assert len(out) == 1
        assert out[0].definition == "Graphics Processing Unit"

    def test_non_alpha_initial_words_are_ignored_in_require_initials_match(self, dummy_cfg):
        cfg = dummy_cfg()
        # Non-alpha-leading words are ignored; we also avoid TitleCase tail trimming
        snippet = "(3M Portable format)"
        out = find_parenthetical_longform_after_acr(snippet, cfg, acr="PF", require_initials_match=True)
        assert len(out) == 1
        assert out[0].definition == "3M Portable format"
        # PF != PDF
        out2 = find_parenthetical_longform_after_acr(snippet, cfg, acr="PDF", require_initials_match=True)
        assert out2 == []

    def test_require_require_initials_match_false_allows_generic_parenthetical(self, dummy_cfg):
        cfg = dummy_cfg()
        snippet = "(see below for details)"
        # Contains letters, normalizes to same text; pass when require_initials_match disabled
        out = find_parenthetical_longform_after_acr(snippet, cfg, acr="ANY", require_initials_match=False)
        assert len(out) == 1
        assert out[0].definition == "see below for details"

    def test_respects_max_phrase_chars(self, dummy_cfg):
        cfg = dummy_cfg(max_phrase_chars=10)
        snippet = "(Hypertext Transfer Protocol)"
        # Longer than max → no match at all
        assert find_parenthetical_longform_after_acr(snippet, cfg, acr="HTTP") == []


class TestFindParentheticalLongformAfterAcrIntegrationEdgesCases:
    def test_nested_parentheses_are_rejected(self, dummy_cfg):
        cfg = dummy_cfg()
        # Regex disallows inner parentheses: [^()]...
        assert find_parenthetical_longform_after_acr("(Portable (Document) Format)", cfg, acr="PDF") == []

    def test_returns_empty_if_normalize_definition_fails_in_bypass_path(self, _patch, dummy_cfg):
        _patch(
            find_parenthetical_longform_after_acr,
            has_letter=lambda s: True,
            tighten_definition_span=lambda s: s,
            normalize_definition=lambda s: "",  # force failure
        )
        cfg = dummy_cfg()
        assert (
            find_parenthetical_longform_after_acr(
                "(Portable Document Format)", cfg, acr="PDF", require_initials_match=False
            )
            == []
        )

    def test_returns_empty_if_normalize_definition_fails_in_initials_path(
        self, _patch, dummy_cfg, hit_cfg, build_stream_seen
    ):
        build_stream_fn, _ = build_stream_seen

        def fake_align(acr, stream, tokens, **kwargs):
            return hit_cfg(tok_left=0, tok_right=len(tokens) - 1, hit_tokens=set(range(len(tokens))))

        _patch(
            find_parenthetical_longform_after_acr,
            has_letter=lambda s: True,
            build_initials_stream=build_stream_fn,
            align_acronym_to_initials=fake_align,
            expand_numeric_leading_window=lambda tokens, i, j: (i, j),
            normalize_definition=lambda s: "",  # force failure
            tighten_definition_span=lambda s: s,
            strip_trailing_punct_str=lambda s: s,
            collapse_ws=lambda s: " ".join(s.split()),
        )

        cfg = dummy_cfg()
        assert (
            find_parenthetical_longform_after_acr(
                "(Portable Document Format)", cfg, acr="PDF", require_initials_match=True
            )
            == []
        )
