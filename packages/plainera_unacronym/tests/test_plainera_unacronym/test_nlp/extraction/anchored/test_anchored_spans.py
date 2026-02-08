
import re

import plainera_unacronym.nlp.extraction.anchored.spans as mod
from plainera_unacronym.nlp.extraction import ExtractionConfig
from plainera_unacronym.nlp.extraction.anchored.spans import _trim_span
from plainera_unacronym.nlp.extraction.matchers.defs.common import LocalDefMatch


class TestTrimSpan:
    def test_no_whitespace_no_change(self):
        seg = "abc"
        assert _trim_span(seg, 0, 3) == (0, 3)

    def test_trims_leading_spaces(self):
        seg = "   abc"
        assert _trim_span(seg, 0, len(seg)) == (3, len(seg))

    def test_trims_trailing_spaces(self):
        seg = "abc   "
        assert _trim_span(seg, 0, len(seg)) == (0, 3)

    def test_trims_both_sides(self):
        seg = " \t abc \n "
        d0, d1 = _trim_span(seg, 0, len(seg))
        assert seg[d0:d1] == "abc"

    def test_does_not_touch_internal_whitespace(self):
        seg = "  a  b  "
        d0, d1 = _trim_span(seg, 0, len(seg))
        assert seg[d0:d1] == "a  b"

    def test_all_whitespace_collapses_to_empty_span(self):
        seg = " \t\n  "
        d0, d1 = _trim_span(seg, 0, len(seg))
        assert (d0, d1) == (len(seg), len(seg))  # fully trimmed to end

    def test_respects_existing_bounds(self):
        seg = "xx  abc  yy"
        # Only trim inside the provided slice
        d0, d1 = _trim_span(seg, 2, 9)  # seg[2:9] == "  abc  "
        assert seg[d0:d1] == "abc"

    def test_empty_slice_is_stable(self):
        seg = "abc"
        assert _trim_span(seg, 1, 1) == (1, 1)

    def test_single_char_whitespace_slice(self):
        seg = "a b"
        d0, d1 = _trim_span(seg, 1, 2)  # slice is " "
        assert (d0, d1) == (2, 2)


def _m_def_before(seg: str, def_text: str, acr: str) -> re.Match[str]:
    """
    Build a minimal match object with groups 'def' and 'acr' spanning within `seg`.
    We use the same group names your function expects.
    """
    pat = re.compile(
        rf"(?P<def>{re.escape(def_text)})\s*\(\s*(?P<acr>{re.escape(acr)})\s*\)"
    )
    m = pat.search(seg)
    assert m is not None
    return m


class TestCalcDefSpanDefBeforeUnit:
    def test_plain_wrapper_uses_helper_matcher_spans(self, _patch):
        cfg = ExtractionConfig()
        seg = "Portable Document Format (PDF)"
        m = _m_def_before(seg, "Portable Document Format", "PDF")

        # Return a real LocalDefMatch using indices relative to `seg`
        loc = LocalDefMatch(
            def_start=m.start("def"),
            def_end=m.end("def"),
            definition="Portable Document Format",
            raw="Portable Document Format",
        )

        _patch(
            mod._calc_def_span_def_before,
            find_parenthetical_longform_before_acr=lambda snippet, acr_norm, cfg: [loc],
        )

        out = mod._calc_def_span_def_before(acr_norm="PDF", seg=seg, m=m, cfg=cfg)
        assert out == (m.start("def"), m.end("def"))

    def test_plain_wrapper_returns_none_when_helper_finds_nothing(self, _patch):
        cfg = ExtractionConfig()
        seg = "Portable Document Format (PDF)"
        m = _m_def_before(seg, "Portable Document Format", "PDF")

        _patch(
            mod._calc_def_span_def_before,
            find_parenthetical_longform_before_acr=lambda *_: [],
        )

        assert mod._calc_def_span_def_before(acr_norm="PDF", seg=seg, m=m, cfg=cfg) is None

    def test_complex_wrapper_bypasses_helper_and_requires_initials_alignment(self, _patch):
        cfg = ExtractionConfig()

        # "quotes around acronym inside wrapper" triggers complex path
        seg = 'Portable Document Format ("PDF")'
        pat = re.compile(r'(?P<def>Portable Document Format)\s*\(\s*"(?P<acr>PDF)"\s*\)')
        m = pat.search(seg)
        assert m is not None

        calls = {"helper": 0}

        def fake_helper(*_a, **_k):
            calls["helper"] += 1
            return [LocalDefMatch(0, 1, "X")]

        _patch(
            mod._calc_def_span_def_before,
            find_parenthetical_longform_before_acr=fake_helper,
            initials_match=lambda acr, phrase: True,  # accept
        )

        out = mod._calc_def_span_def_before(acr_norm="PDF", seg=seg, m=m, cfg=cfg)

        # Complex branch: helper must not be called
        assert calls["helper"] == 0
        # Should return the trimmed m.span("def")
        assert out == (m.start("def"), m.end("def"))

    def test_complex_wrapper_rejects_when_initials_do_not_match(self, _patch):
        cfg = ExtractionConfig()
        seg = "Lots Of Llamas (LOL, etc.)"
        # Tail punctuation triggers complex path (comma)
        pat = re.compile(r"(?P<def>Lots Of Llamas)\s*\(\s*(?P<acr>LOL)\s*,\s*etc\.\s*\)")
        m = pat.search(seg)
        assert m is not None

        _patch(
            mod._calc_def_span_def_before,
            initials_match=lambda acr, phrase: False,  # reject
        )

        assert mod._calc_def_span_def_before(acr_norm="LOL", seg=seg, m=m, cfg=cfg) is None


class TestCalcDefSpanDefBeforeIntegration:
    def test_complex_tail_keeps_only_def_group_span_and_trims_ws(self, _patch):
        cfg = ExtractionConfig()
        seg = "  Portable Document Format   (PDF, including forms)  "
        # Make a match where def group includes extra surrounding whitespace
        pat = re.compile(r"\s*(?P<def>Portable Document Format)\s*\(\s*(?P<acr>PDF)\s*,\s*including forms\s*\)\s*")
        m = pat.search(seg)
        assert m is not None

        # Let initials_match be real if you want; we’ll keep it deterministic here.
        _patch(mod._calc_def_span_def_before, initials_match=lambda acr, phrase: True)

        d0, d1 = mod._calc_def_span_def_before(acr_norm="PDF", seg=seg, m=m, cfg=cfg)
        assert seg[d0:d1] == "Portable Document Format"


class TestCalcDefSpanInlineAfterUnit:
    def test_returns_none_when_matcher_returns_empty(self, _patch):
        cfg = ExtractionConfig()
        seg = "NLP stands for Natural language processing."
        acr_end_local = seg.index("NLP") + len("NLP")

        _patch(
            mod._calc_def_span_inline_after,
            find_inline_longform_after_acr=lambda *_a, **_k: [],
        )

        assert (
            mod._calc_def_span_inline_after(
                acr_norm="NLP",
                seg=seg,
                acr_end_local=acr_end_local,
                cfg=cfg,
            )
            is None
        )

    def test_rebases_matcher_span_into_seg_offsets(self, _patch):
        cfg = ExtractionConfig()
        seg = "NLP stands for Natural language processing."
        acr_end_local = seg.index("NLP") + len("NLP")

        # The matcher sees only snippet = seg[acr_end_local:].
        snippet = seg[acr_end_local:]

        # We want the matcher to return def spans *relative to snippet*.
        # Choose the exact substring "Natural language processing".
        def_text = "Natural language processing"
        rel_start = snippet.index(def_text)
        rel_end = rel_start + len(def_text)

        loc = LocalDefMatch(def_start=rel_start, def_end=rel_end, definition=def_text, raw=def_text)

        _patch(
            mod._calc_def_span_inline_after,
            find_inline_longform_after_acr=lambda snippet_arg,
                                                  cfg_arg,
                                                  acr,
                                                  *,
                                                  max_chars,
                                                  require_initials_match: [loc],
        )

        out = mod._calc_def_span_inline_after(
            acr_norm="NLP",
            seg=seg,
            acr_end_local=acr_end_local,
            cfg=cfg,
        )
        assert out == (acr_end_local + rel_start, acr_end_local + rel_end)
        assert seg[out[0] : out[1]] == def_text

    def test_passes_expected_flags_and_max_chars_to_matcher(self, _patch):
        cfg = ExtractionConfig(max_phrase_chars=123)
        seg = "NLP stands for Natural language processing."
        acr_end_local = seg.index("NLP") + len("NLP")

        calls = {}

        def fake_matcher(snippet, cfg_arg, acr, *, max_chars=None, require_initials_match=True):
            calls["snippet"] = snippet
            calls["cfg"] = cfg_arg
            calls["acr"] = acr
            calls["max_chars"] = max_chars
            calls["require_initials_match"] = require_initials_match
            return []

        _patch(mod._calc_def_span_inline_after, find_inline_longform_after_acr=fake_matcher)

        mod._calc_def_span_inline_after(acr_norm="NLP", seg=seg, acr_end_local=acr_end_local, cfg=cfg)

        assert calls["snippet"] == seg[acr_end_local:]
        assert calls["cfg"] is cfg
        assert calls["acr"] == "NLP"
        assert calls["max_chars"] == cfg.max_phrase_chars * 2
        assert calls["require_initials_match"] is True


class TestCalcDefSpanInlineAfterIntegration:
    def test_span_points_into_seg_even_with_leading_space_after_acr(self, _patch):
        cfg = ExtractionConfig()
        seg = "NLP  stands for Natural language processing."
        acr_end_local = seg.index("NLP") + len("NLP")

        d0, d1 = mod._calc_def_span_inline_after(acr_norm="NLP", seg=seg, acr_end_local=acr_end_local, cfg=cfg)
        assert seg[d0:d1] == "Natural language processing."


class TestCalcDefSpanDefAfterUnit:
    def test_returns_none_when_matcher_returns_empty(self, _patch):
        cfg = ExtractionConfig()
        seg = "SSO (Single sign-on) is enabled."
        acr_end_local = seg.index("SSO") + len("SSO")

        _patch(
            mod._calc_def_span_def_after,
            find_parenthetical_longform_after_acr=lambda *_a, **_k: [],
        )

        assert (
            mod._calc_def_span_def_after(
                acr_norm="SSO",
                seg=seg,
                acr_end_local=acr_end_local,
                cfg=cfg,
            )
            is None
        )

    def test_rebases_span_without_joiner(self, _patch):
        cfg = ExtractionConfig()
        seg = "SSO (Single sign-on) is enabled."
        acr_end_local = seg.index("SSO") + len("SSO")

        snippet = seg[acr_end_local:]
        j = mod._POSSESSIVE_JOIN_RE.match(snippet)
        join_off = j.end() if j else 0
        snippet2 = snippet[join_off:]  # <-- this is what the prod code uses

        def_text = "Single sign-on"
        rel_start = snippet2.index(def_text)
        rel_end = rel_start + len(def_text)
        loc = LocalDefMatch(def_start=rel_start, def_end=rel_end, definition=def_text, raw=def_text)

        _patch(
            mod._calc_def_span_def_after,
            find_parenthetical_longform_after_acr=lambda *_a, **_k: [loc],
        )

        d0, d1 = mod._calc_def_span_def_after(acr_norm="SSO", seg=seg, acr_end_local=acr_end_local, cfg=cfg)
        assert seg[d0:d1] == def_text

    def test_rebases_span_with_possessive_joiner(self, _patch):
        cfg = ExtractionConfig()
        seg = "PDF's (Portable Document Format) is common."
        acr_end_local = seg.index("PDF") + len("PDF")

        snippet = seg[acr_end_local:]  # starts with "'s (Portable..."
        j = mod._POSSESSIVE_JOIN_RE.match(snippet)
        assert j is not None  # sanity
        join_off = j.end()

        snippet2 = snippet[join_off:]
        def_text = "Portable Document Format"
        rel_start = snippet2.index(def_text)
        rel_end = rel_start + len(def_text)
        loc = LocalDefMatch(def_start=rel_start, def_end=rel_end, definition=def_text, raw=def_text)

        _patch(
            mod._calc_def_span_def_after,
            find_parenthetical_longform_after_acr=lambda *_a, **_k: [loc],
        )

        d0, d1 = mod._calc_def_span_def_after(acr_norm="PDF", seg=seg, acr_end_local=acr_end_local, cfg=cfg)
        assert (d0, d1) == (acr_end_local + join_off + rel_start, acr_end_local + join_off + rel_end)
        assert seg[d0:d1] == def_text

    def test_rebases_span_with_punct_joiner(self, _patch):
        cfg = ExtractionConfig()
        seg = "PPE, (Personal Protective Equipment) is required."
        acr_end_local = seg.index("PPE") + len("PPE")

        snippet = seg[acr_end_local:]  # starts with ", (Personal..."
        j = mod._POSSESSIVE_JOIN_RE.match(snippet)
        assert j is not None
        join_off = j.end()

        snippet2 = snippet[join_off:]
        def_text = "Personal Protective Equipment"
        rel_start = snippet2.index(def_text)
        rel_end = rel_start + len(def_text)
        loc = LocalDefMatch(def_start=rel_start, def_end=rel_end, definition=def_text, raw=def_text)

        _patch(
            mod._calc_def_span_def_after,
            find_parenthetical_longform_after_acr=lambda *_a, **_k: [loc],
        )

        d0, d1 = mod._calc_def_span_def_after(acr_norm="PPE", seg=seg, acr_end_local=acr_end_local, cfg=cfg)
        assert seg[d0:d1] == def_text

    def test_passes_expected_flags_to_matcher(self, _patch):
        cfg = ExtractionConfig()
        seg = "SSO (Single sign-on) is enabled."
        acr_end_local = seg.index("SSO") + len("SSO")

        calls = {}

        def fake_matcher(snippet2, cfg_arg, acr, *, require_initials_match=True):
            calls["snippet2"] = snippet2
            calls["cfg"] = cfg_arg
            calls["acr"] = acr
            calls["require_initials_match"] = require_initials_match
            return []

        _patch(mod._calc_def_span_def_after, find_parenthetical_longform_after_acr=fake_matcher)

        mod._calc_def_span_def_after(acr_norm="SSO", seg=seg, acr_end_local=acr_end_local, cfg=cfg)

        snippet = seg[acr_end_local:]
        j = mod._POSSESSIVE_JOIN_RE.match(snippet)
        join_off = j.end() if j else 0
        assert calls["snippet2"] == snippet[join_off:]
        assert calls["cfg"] is cfg
        assert calls["acr"] == "SSO"
        assert calls["require_initials_match"] is True


class TestCalcDefSpanDefAfterIntegration:
    def test_acr_then_paren_extracts_longform_span(self):
        cfg = ExtractionConfig()
        seg = "SSO (Single sign-on) is enabled."
        acr_end_local = seg.index("SSO") + len("SSO")

        span = mod._calc_def_span_def_after(
            acr_norm="SSO",
            seg=seg,
            acr_end_local=acr_end_local,
            cfg=cfg,
        )
        assert span is not None
        d0, d1 = span
        assert seg[d0:d1] == "Single sign-on"

    def test_possessive_joiner_is_skipped_and_span_is_correct(self):
        cfg = ExtractionConfig()
        seg = "PDF's (Portable Document Format) is common."
        acr_end_local = seg.index("PDF") + len("PDF")

        span = mod._calc_def_span_def_after(
            acr_norm="PDF",
            seg=seg,
            acr_end_local=acr_end_local,
            cfg=cfg,
        )
        assert span is not None
        d0, d1 = span
        assert seg[d0:d1] == "Portable Document Format"
