from document_resolution.nlp.common.types import ExtractedDefinition, FirstOccurrence
from document_resolution.nlp.extraction.acronyms.backref.extract import (
    _candidate_from_prev_sentence,
    _find_backref_candidate,
    _score_backref_confidence,
    extract_sentence_backrefs,
    _valid_backref_candidate,
)
from document_resolution.nlp.extraction.acronyms.config import ExtractionConfig


class TestValidBackrefCandidate:
    def test_rejects_empty(self, _patch):
        _patch(
            _valid_backref_candidate,
            initials_match=lambda *_a, **_k: True,
            _initials_match_backref=lambda *_a, **_k: True,
        )

        assert (
            _valid_backref_candidate(
                clean="",
                acr_norm="SSO",
                max_chars=200,
                require_two_words=True,
            )
            is False
        )

    def test_rejects_over_max_chars(self, _patch):
        _patch(
            _valid_backref_candidate,
            initials_match=lambda *_a, **_k: True,
            _initials_match_backref=lambda *_a, **_k: True,
        )

        assert (
            _valid_backref_candidate(
                clean="x" * 201,
                acr_norm="SSO",
                max_chars=200,
                require_two_words=False,
            )
            is False
        )

    def test_rejects_candidate_equal_to_acronym_ignoring_spaces_and_case(self, _patch):
        _patch(
            _valid_backref_candidate,
            initials_match=lambda *_a, **_k: True,
            _initials_match_backref=lambda *_a, **_k: True,
        )

        assert (
            _valid_backref_candidate(
                clean="s s o",
                acr_norm="SSO",
                max_chars=200,
                require_two_words=False,
            )
            is False
        )

    def test_requires_two_words_when_enabled(self, _patch):
        _patch(
            _valid_backref_candidate,
            initials_match=lambda *_a, **_k: True,
            _initials_match_backref=lambda *_a, **_k: True,
        )

        assert (
            _valid_backref_candidate(
                clean="Single",
                acr_norm="S",
                max_chars=200,
                require_two_words=True,
            )
            is False
        )

    def test_accepts_when_strict_initials_match_passes(self, _patch):
        _patch(
            _valid_backref_candidate,
            initials_match=lambda *_a, **_k: True,
            _initials_match_backref=lambda *_a, **_k: False,
        )

        assert (
            _valid_backref_candidate(
                clean="Single sign on",
                acr_norm="SSO",
                max_chars=200,
                require_two_words=True,
            )
            is True
        )

    def test_accepts_when_hyphen_aware_fallback_passes_even_if_strict_fails(self, _patch):
        _patch(
            _valid_backref_candidate,
            initials_match=lambda *_a, **_k: False,
            _initials_match_backref=lambda *_a, **_k: True,
        )

        assert (
            _valid_backref_candidate(
                clean="Single sign-on",
                acr_norm="SSO",
                max_chars=200,
                require_two_words=True,
            )
            is True
        )

    def test_rejects_when_both_initials_matchers_fail(self, _patch):
        _patch(
            _valid_backref_candidate,
            initials_match=lambda *_a, **_k: False,
            _initials_match_backref=lambda *_a, **_k: False,
        )

        assert (
            _valid_backref_candidate(
                clean="Single sign-on",
                acr_norm="SSO",
                max_chars=200,
                require_two_words=True,
            )
            is False
        )


class TestScoreBackrefConfidence:
    def test_score_backref_confidence_penalises_lookback_and_distance(self):
        cfg = ExtractionConfig()  # with confidence defaults
        c1, _ = _score_backref_confidence(
            cfg=cfg, fo_surface="SSO", cand="Single Sign-on", evidence="definitionish", back=1, dist_chars=5
        )
        c2, _ = _score_backref_confidence(
            cfg=cfg, fo_surface="SSO", cand="Single Sign-on", evidence="definitionish", back=2, dist_chars=5
        )
        assert c1 > c2


class TestCandidateFromPrevSentenceIntegration:
    def test_happy_path_returns_normalised_candidate(self):
        cfg = ExtractionConfig()
        prev = "We use Single sign-on for authentication."
        out, evidence = _candidate_from_prev_sentence(
            acr_norm="SSO",
            prev_text=prev,
            cfg=cfg,
            max_chars=200,
            require_two_words=True,
        )
        assert out == "Single sign-on"
        assert evidence == "definitionish"

    def test_returns_none_for_blank_prev_text(self):
        cfg = ExtractionConfig()
        assert (
            _candidate_from_prev_sentence(
                acr_norm="SSO",
                prev_text="   \n\t",
                cfg=cfg,
                max_chars=200,
                require_two_words=True,
            )
            is None
        )

    def test_filters_out_very_long_prev_sentence(self):
        cfg = ExtractionConfig()
        prev = "Word " * 1000  # collapsed length likely > max_chars*3
        assert (
            _candidate_from_prev_sentence(
                acr_norm="SSO",
                prev_text=prev,
                cfg=cfg,
                max_chars=50,
                require_two_words=True,
            )
            is None
        )

    def test_strips_trailing_punctuation_before_span_search(self):
        cfg = ExtractionConfig()
        prev = "We use Single sign-on for authentication...   "
        out, evidence = _candidate_from_prev_sentence(
            acr_norm="SSO",
            prev_text=prev,
            cfg=cfg,
            max_chars=200,
            require_two_words=True,
        )
        assert out == "Single sign-on"
        assert evidence == "definitionish"

    def test_returns_none_when_no_initials_span_found(self):
        cfg = ExtractionConfig()
        prev = "Nothing matching here."
        assert (
            _candidate_from_prev_sentence(
                acr_norm="SSO",
                prev_text=prev,
                cfg=cfg,
                max_chars=200,
                require_two_words=True,
            )
            is None
        )

    def test_rejects_candidate_without_letters(self):
        cfg = ExtractionConfig()
        # This sentence can never produce a valid initials span with letters for "A"
        prev = "123 456 789."
        assert (
            _candidate_from_prev_sentence(
                acr_norm="A",
                prev_text=prev,
                cfg=cfg,
                max_chars=200,
                require_two_words=False,
            )
            is None
        )

    def test_rejects_candidate_equal_to_acronym(self):
        cfg = ExtractionConfig()
        prev = "We use SSO for authentication."
        assert (
            _candidate_from_prev_sentence(
                acr_norm="SSO",
                prev_text=prev,
                cfg=cfg,
                max_chars=200,
                require_two_words=False,
            )
            is None
        )

    def test_rejects_candidate_over_max_chars(self):
        cfg = ExtractionConfig()
        prev = "We use Single sign-on for authentication."
        assert (
            _candidate_from_prev_sentence(
                acr_norm="SSO",
                prev_text=prev,
                cfg=cfg,
                max_chars=5,  # far too small for "Single sign-on"
                require_two_words=False,
            )
            is None
        )

    def test_initials_match_validation_blocks_false_positive(self):
        cfg = ExtractionConfig()
        # "Lots Of Llamas" gives initials LOL; acronym LLO should not validate
        prev = "We use Lots Of Llamas in testing."
        assert (
            _candidate_from_prev_sentence(
                acr_norm="LLO",
                prev_text=prev,
                cfg=cfg,
                max_chars=200,
                require_two_words=True,
            )
            is None
        )


class TestFindBackrefCandidate:
    def test_returns_none_when_no_prev_sentences(self, _patch):
        cfg = ExtractionConfig()
        text = "Only one sentence."
        spans = [(0, len(text))]
        si = 0

        # Should never call candidate finder because si==0 gives empty loop
        called = {"n": 0}

        def fake_candidate(**_):
            called["n"] += 1
            return "X"

        _patch(_find_backref_candidate, _candidate_from_prev_sentence=fake_candidate)

        out = _find_backref_candidate(
            text=text,
            spans=spans,
            si=si,
            acr_norm="SSO",
            cfg=cfg,
            max_chars=200,
            require_two_words=True,
        )
        assert out is None
        assert called["n"] == 0

    def test_stops_at_first_hit_nearest_prev_sentence(self, _patch):
        text = "S0. S1. S2."
        spans = [(0, 3), (4, 7), (8, 11)]
        si = 2
        cfg = ExtractionConfig()

        seen_prev_texts: list[str] = []

        def fake_candidate(*, prev_text, **_):
            seen_prev_texts.append(prev_text)
            return ("HIT", "definitionish") if prev_text.strip() == "S1." else None

        _patch(_find_backref_candidate, _candidate_from_prev_sentence=fake_candidate)

        out = _find_backref_candidate(
            text=text,
            spans=spans,
            si=si,
            acr_norm="SSO",
            cfg=cfg,
            max_chars=200,
            require_two_words=True,
        )

        assert out == ("HIT", spans[1], 1, "definitionish")
        assert seen_prev_texts == ["S1."]

    def test_falls_back_to_older_sentence_if_nearest_has_no_hit(self, _patch):
        text = "Alpha.\nBeta.\nGamma."
        spans = [(0, 6), (7, 12), (13, 19)]
        si = 2
        cfg = ExtractionConfig()

        calls: list[str] = []

        def fake_candidate(*, prev_text, **_):
            calls.append(prev_text.strip())
            if prev_text.strip() == "Alpha.":
                return "FOUND", "definitionish"
            return None

        _patch(_find_backref_candidate, _candidate_from_prev_sentence=fake_candidate)

        out = _find_backref_candidate(
            text=text,
            spans=spans,
            si=si,
            acr_norm="SSO",
            cfg=cfg,
            max_chars=200,
            require_two_words=True,
        )

        assert out == ("FOUND", spans[0], 2, "definitionish")
        # Must attempt Beta first (nearest), then Alpha
        assert calls == ["Beta.", "Alpha."]

    def test_respects_sentence_backref_lookback_limit(self, _patch):
        text = "S0. S1. S2. S3."
        spans = [(0, 3), (4, 7), (8, 11), (12, 15)]
        si = 3
        cfg = ExtractionConfig()
        # cfg is frozen+slots; use object.__setattr__ to override for tests
        object.__setattr__(cfg, "sentence_backref_lookback", 1)

        calls: list[str] = []

        def fake_candidate(*, prev_text, **_):
            calls.append(prev_text)
            return None

        _patch(_find_backref_candidate, _candidate_from_prev_sentence=fake_candidate)

        out = _find_backref_candidate(
            text=text,
            spans=spans,
            si=si,
            acr_norm="SSO",
            cfg=cfg,
            max_chars=200,
            require_two_words=True,
        )
        assert out is None
        # Only the nearest previous span should be tried
        assert calls == ["S2."]  # only nearest previous sentence checked

    def test_forwards_flags_and_args(self, _patch):
        text = "Prev. Here."
        spans = [(0, 5), (6, 11)]
        si = 1
        cfg = ExtractionConfig()

        seen = {}

        def fake_candidate(**kwargs):
            seen.update(kwargs)
            return "OK", "definitionish"

        _patch(_find_backref_candidate, _candidate_from_prev_sentence=fake_candidate)

        out = _find_backref_candidate(
            text=text,
            spans=spans,
            si=si,
            acr_norm="SSO",
            cfg=cfg,
            max_chars=123,
            require_two_words=False,
        )

        assert out == ("OK", spans[0], 1, "definitionish")
        assert seen["acr_norm"] == "SSO"
        assert seen["cfg"] is cfg
        assert seen["max_chars"] == 123
        assert seen["require_two_words"] is False
        assert seen["prev_text"] == "Prev."


def test_sentence_backref_ignores_single_letter_acronyms():
    cfg = ExtractionConfig()
    text = "We use Authentication. A is sometimes used as shorthand."
    firsts = {
        "A": FirstOccurrence(
            acronym="A",
            start_offset=text.index("A is"),
            end_offset=text.index("A is") + 1,
            occurrence_confidence=0.9,
            normalized_key="A",
        )
    }

    out = extract_sentence_backrefs(text=text, firsts=firsts, cfg=cfg)
    assert out == []


def _fo(acr: str, start: int, end: int, *, norm: str | None = None):
    from document_resolution.nlp.common.types import FirstOccurrence

    return FirstOccurrence(
        acronym=acr,
        start_offset=start,
        end_offset=end,
        occurrence_confidence=0.9,
        normalized_key=norm,
    )


class TestExtractSentenceBackrefsUnit:
    def test_returns_empty_when_no_spans(self, _patch):
        cfg = ExtractionConfig()

        def fake_sent_spans(_text):
            return []

        _patch(extract_sentence_backrefs, sent_spans=fake_sent_spans)

        out = extract_sentence_backrefs(text="Anything.", firsts={}, cfg=cfg)
        assert out == []

    def test_skips_when_acronym_in_first_sentence(self, _patch):
        cfg = ExtractionConfig()
        text = "SSO appears here. And later."
        # spans: [0..end_of_first], [start_second..end]
        spans = [(0, text.index(".") + 1), (text.index("And"), len(text))]

        firsts = {"SSO": _fo("SSO", start=text.index("SSO"), end=text.index("SSO") + 3, norm="SSO")}

        _patch(
            extract_sentence_backrefs,
            sent_spans=lambda _t: spans,
            find_span_index=lambda _spans, _pos: 0,  # force "first sentence"
        )

        # Should not try to find candidates at all if si==0
        called = {"n": 0}

        def fake_find(**_):
            called["n"] += 1
            return "X", spans[0]

        _patch(extract_sentence_backrefs, _find_backref_candidate=fake_find)

        out = extract_sentence_backrefs(text=text, firsts=firsts, cfg=cfg)
        assert out == []
        assert called["n"] == 0

    def test_skips_single_letter_acronyms_via_min_acr_len(self, _patch):
        cfg = ExtractionConfig()  # min_acr_len defaults to 2
        text = "We use Authentication. A is used later."

        spans = [(0, text.index(".") + 1), (text.index("A"), len(text))]
        firsts = {"A": _fo("A", start=text.index("A"), end=text.index("A") + 1, norm="A")}

        _patch(
            extract_sentence_backrefs,
            sent_spans=lambda _t: spans,
            find_span_index=lambda _spans, _pos: 1,
        )

        called = {"n": 0}

        def fake_find(**_):
            called["n"] += 1
            return "Authentication", spans[0]

        _patch(extract_sentence_backrefs, _find_backref_candidate=fake_find)

        out = extract_sentence_backrefs(text=text, firsts=firsts, cfg=cfg)
        assert out == []
        assert called["n"] == 0  # never called due to min length gate

    def test_emits_definition_when_candidate_found(self, _patch):
        cfg = ExtractionConfig()
        text = "Prev sentence. Later SSO appears."
        spans = [(0, text.index(".") + 1), (text.index("Later"), len(text))]

        fo = _fo("SSO", start=text.index("SSO"), end=text.index("SSO") + 3, norm="SSO")
        firsts = {"SSO": fo}

        emitted: list[dict] = []

        def fake_emit(*, acr_norm, fo, cand, prev_span, text, cfg, back, evidence):
            emitted.append(
                {
                    "acr_norm": acr_norm,
                    "cand": cand,
                    "prev_span": prev_span,
                    "fo": fo,
                    "text": text,
                    "back": back,
                    "evidence": evidence,
                }
            )
            return ExtractedDefinition(
                acronym=acr_norm,
                definition=cand,
                source="backref",
                definition_confidence=0.77,
                acr_start=fo.start_offset,
                acr_end=fo.end_offset,
                def_start=prev_span[0],
                def_end=prev_span[1],
                original_definition=cand,
                kind="sentence_backref",
            )

        _patch(
            extract_sentence_backrefs,
            sent_spans=lambda _t: spans,
            find_span_index=lambda _spans, _pos: 1,
            _find_backref_candidate=lambda **_: ("Single sign-on", spans[0], 1, "definitionish"),
            _emit_backref_def=fake_emit,
        )

        out = extract_sentence_backrefs(text=text, firsts=firsts, cfg=cfg)
        assert len(out) == 1
        assert out[0].acronym == "SSO"
        assert out[0].definition == "Single sign-on"
        assert emitted and emitted[0]["prev_span"] == spans[0]
        assert emitted[0]["acr_norm"] == "SSO"

    def test_multiple_firsts_only_emits_for_hits_preserves_firsts_iteration_order(self, _patch):
        cfg = ExtractionConfig()
        text = "Prev. Later SSO. Later GPU."
        spans = [(0, text.index(".") + 1), (text.index("Later"), len(text))]

        sso_fo = _fo("SSO", start=text.index("SSO"), end=text.index("SSO") + 3, norm="SSO")
        gpu_fo = _fo("GPU", start=text.index("GPU"), end=text.index("GPU") + 3, norm="GPU")

        # insertion order: SSO then GPU
        firsts = {"SSO": sso_fo, "GPU": gpu_fo}

        def fake_find(*, acr_norm, **_):
            if acr_norm == "SSO":
                return "Single sign-on", spans[0], 1, "definitionish"
            return None

        def fake_emit(*, acr_norm, fo, cand, prev_span, text, cfg, back, evidence):
            return ExtractedDefinition(
                acronym=acr_norm,
                definition=cand,
                source="backref",
                definition_confidence=0.77,
                acr_start=fo.start_offset,
                acr_end=fo.end_offset,
                def_start=prev_span[0],
                def_end=prev_span[1],
                original_definition=cand,
                kind="sentence_backref",
            )

        _patch(
            extract_sentence_backrefs,
            _emit_backref_def=fake_emit,
            _find_backref_candidate=fake_find,
            sent_spans=lambda _t: spans,
            find_span_index=lambda _spans, _pos: 1,
        )

        out = extract_sentence_backrefs(text=text, firsts=firsts, cfg=cfg)
        assert [d.acronym for d in out] == ["SSO"]


def _span(text: str, needle: str) -> tuple[int, int]:
    i = text.index(needle)
    return i, i + len(needle)


class TestExtractSentenceBackrefsIntegration:
    def test_happy_path_sso_previous_sentence(self):
        cfg = ExtractionConfig()
        # object.__setattr__(cfg, "sentence_backref_require_two_words", True)

        text = "We use Single sign-on for authentication. This is known as SSO."
        a0, a1 = _span(text, "SSO")
        firsts = {"SSO": _fo("SSO", a0, a1, norm="SSO")}

        out = extract_sentence_backrefs(text=text, firsts=firsts, cfg=cfg)
        assert len(out) == 1
        d = out[0]
        assert d.acronym == "SSO"
        assert d.definition == "Single sign-on"
        assert d.source == "sentence_backref"
        assert d.kind == "sentence_backref"
        # sanity: acronym span maps from FO
        assert (d.acr_start, d.acr_end) == (a0, a1)

    def test_does_not_fire_when_acronym_in_first_sentence(self):
        cfg = ExtractionConfig()
        text = "SSO is enabled. We use Single sign-on for authentication."
        a0, a1 = _span(text, "SSO")
        firsts = {"SSO": _fo("SSO", a0, a1, norm="SSO")}

        out = extract_sentence_backrefs(text=text, firsts=firsts, cfg=cfg)
        assert out == []

    def test_respects_lookback_limit(self):
        cfg = ExtractionConfig()
        object.__setattr__(cfg, "sentence_backref_lookback", 1)

        text = "We use Single sign-on for authentication. Nothing here. This is known as SSO."
        a0, a1 = _span(text, "SSO")
        firsts = {"SSO": _fo("SSO", a0, a1, norm="SSO")}

        out = extract_sentence_backrefs(text=text, firsts=firsts, cfg=cfg)
        # With lookback=1, only "Nothing here." is searched, so no hit.
        assert out == []
