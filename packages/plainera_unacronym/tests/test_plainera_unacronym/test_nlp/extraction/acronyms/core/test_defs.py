from types import SimpleNamespace as NS

import plainera_unacronym.nlp.extraction.acronyms.core.defs as mod
import pytest
from plainera_unacronym.nlp.common.types import (
    ExtractedDefinition,
    InTextPick,  # noqa: E402
    Span,
)
from plainera_unacronym.nlp.extraction.acronyms.backref.extract import (
    _score_backref_confidence,
    _valid_backref_candidate,
)
from plainera_unacronym.nlp.extraction.acronyms.core.defs import _meaning_key, dedupe_defs, defs_from_picks


def _span(text: str, needle: str) -> Span:
    i = text.index(needle)
    return i, i + len(needle)


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


def _cfg(
    *,
    base: float = 0.60,
    backref_definitionish_boost: float = 0.10,
    backref_initials_boost: float = 0.00,
    backref_lookback_penalty: float = 0.05,
    backref_distance_penalty_per_char: float = 0.0005,
    backref_distance_penalty_cap_chars: int = 200,
    backref_uppercase_acronym_boost: float = 0.05,
    backref_titlecase_ratio_threshold: float = 0.80,
    backref_titlecase_boost: float = 0.05,
):
    conf = NS(
        base_by_source={"sentence_backref": base},
        backref_definitionish_boost=backref_definitionish_boost,
        backref_initials_boost=backref_initials_boost,
        backref_lookback_penalty=backref_lookback_penalty,
        backref_distance_penalty_per_char=backref_distance_penalty_per_char,
        backref_distance_penalty_cap_chars=backref_distance_penalty_cap_chars,
        backref_uppercase_acronym_boost=backref_uppercase_acronym_boost,
        backref_titlecase_ratio_threshold=backref_titlecase_ratio_threshold,
        backref_titlecase_boost=backref_titlecase_boost,
    )
    return NS(confidence=conf)


class TestScoreBackrefConfidenceUnit:
    def test_definitionish_nearest_applies_definitionish_boost_only(self):
        cfg = _cfg(base=0.60)
        score, reasons = _score_backref_confidence(
            cfg=cfg,
            fo_surface="Sso",  # not all-caps -> no acr_caps boost
            cand="single sign-on",  # titlecase ratio low -> no titlecase boost
            evidence="definitionish",
            back=1,
            dist_chars=0,
        )

        assert score == pytest.approx(0.70)
        assert reasons[0] == "base=0.6000"
        assert "evidence=definitionish:+0.1000" in reasons
        assert "final=0.7000" in reasons

    def test_initials_lookback_penalty_accumulates(self):
        cfg = _cfg(base=0.60, backref_lookback_penalty=0.05)
        score, reasons = _score_backref_confidence(
            cfg=cfg,
            fo_surface="sso",
            cand="single sign on",
            evidence="initials",
            back=3,  # penalty = (3-1)*0.05 = 0.10
            dist_chars=0,
        )

        assert score == pytest.approx(0.50)
        assert "base=0.6000" in reasons
        assert "evidence=initials:0" in reasons
        assert "lookback=3:-0.1000" in reasons
        assert "final=0.5000" in reasons

    def test_distance_penalty_is_capped(self):
        cfg = _cfg(base=0.60, backref_distance_penalty_per_char=0.0005, backref_distance_penalty_cap_chars=200)
        score, reasons = _score_backref_confidence(
            cfg=cfg,
            fo_surface="sso",
            cand="single sign on",
            evidence="initials",
            back=1,
            dist_chars=10_000,  # cap at 200 -> penalty 0.1
        )

        assert score == pytest.approx(0.50)
        assert "dist_chars=200:-0.1000" in reasons
        assert "final=0.5000" in reasons

    def test_uppercase_acronym_boost_applies(self):
        cfg = _cfg(base=0.60, backref_uppercase_acronym_boost=0.05)
        score, reasons = _score_backref_confidence(
            cfg=cfg,
            fo_surface="SSO",  # all-caps -> boost
            cand="single sign on",
            evidence="initials",
            back=1,
            dist_chars=0,
        )

        assert score == pytest.approx(0.65)
        assert "acr_caps:+0.0500" in reasons
        assert "final=0.6500" in reasons

    def test_titlecase_boost_applies_when_ratio_meets_threshold(self):
        cfg = _cfg(base=0.60, backref_titlecase_ratio_threshold=0.80, backref_titlecase_boost=0.05)
        score, reasons = _score_backref_confidence(
            cfg=cfg,
            fo_surface="sso",
            cand="Single Sign On",  # 3/3 titlecased -> ratio 1.0
            evidence="initials",
            back=1,
            dist_chars=0,
        )

        assert score == pytest.approx(0.65)
        assert "titlecase=1.00:+0.0500" in reasons
        assert "final=0.6500" in reasons

    def test_clamps_to_point_nine_nine(self):
        cfg = _cfg(base=0.98, backref_definitionish_boost=0.10, backref_uppercase_acronym_boost=0.05)
        score, reasons = _score_backref_confidence(
            cfg=cfg,
            fo_surface="SSO",
            cand="Single Sign On",
            evidence="definitionish",
            back=1,
            dist_chars=0,
        )

        assert score == pytest.approx(0.99)
        assert reasons[-1] == "final=0.9900"


class TestDefsFromPicks:
    def test_returns_empty_for_empty_input(self, monkeypatch):
        # Tighten shouldn't be called, but keep deterministic just in case
        monkeypatch.setattr(mod, "tighten_label_by_acronym", lambda *a, **k: "N/A", raising=True)
        assert mod.defs_from_picks("", {}) == []

    def test_skips_none_and_maps_fields(self, monkeypatch):
        calls: list[tuple[str, str]] = []

        def fake_tighten(raw_label: str, acronym: str, **_):
            calls.append((raw_label, acronym))
            return f"TIGHT[{raw_label}|{acronym}]"

        monkeypatch.setattr(mod, "tighten_label_by_acronym", fake_tighten, raising=True)

        text = "Please turn over (PTO)."
        long = "Please turn over"
        a0, a1 = _span(text, "PTO")
        d0, d1 = _span(text, long)

        picks: dict[str, InTextPick | None] = {
            "good": InTextPick(
                definition=long,
                acr_span=(a0, a1),
                def_span=(d0, d1),
                definition_confidence=0.91,
                original_definition=long,
                route="all_occ_scan_parenthetical",
            ),
            "skip": None,
        }

        out = defs_from_picks(text, picks)
        assert len(out) == 1
        item = out[0]

        assert item.source == "all_occ_scan_parenthetical"
        assert item.acronym == "PTO"  # uppercased
        assert item.definition == "TIGHT[Please turn over|PTO]"
        assert item.original_definition == long
        assert (item.acr_start, item.acr_end) == (a0, a1)
        assert (item.def_start, item.def_end) == (d0, d1)
        assert item.definition_confidence == pytest.approx(0.91)

        # tighten called with UPPER acronym
        assert calls == [(long, "PTO")]

    def test_mixed_case_surface_is_uppercased(self, monkeypatch):
        seen: dict[str, tuple[str, str]] = {}

        def fake_tighten(raw_label: str, acronym: str, **_):
            seen["args"] = (raw_label, acronym)
            return raw_label

        monkeypatch.setattr(mod, "tighten_label_by_acronym", fake_tighten, raising=True)

        text = "Look up Pto later."
        acr_surface = "Pto"
        long = "Please turn over"
        a0, a1 = _span(text, acr_surface)

        pick = InTextPick(
            definition=long,
            acr_span=(a0, a1),
            def_span=(0, len(long)),
            definition_confidence=0.5,
            original_definition=long,
        )

        out = defs_from_picks(text, {"only": pick})
        assert len(out) == 1
        assert out[0].acronym == "PTO"
        assert seen["args"] == (long, "PTO")

    def test_multiple_picks_preserve_insertion_order(self, monkeypatch):
        monkeypatch.setattr(
            mod,
            "tighten_label_by_acronym",
            lambda s, a, **_: f"{s}<{a}>",
            raising=True,
        )

        text = "… PTO … then PoM …"
        a0, a1 = _span(text, "PTO")
        b0, b1 = _span(text, "PoM")

        p1 = InTextPick("Please turn over", (a0, a1), (0, 3), 0.9, "Please turn over")
        p2 = InTextPick("Portable Monitor", (b0, b1), (7, 15), 0.6, "Portable Monitor")

        picks = {"first": p1, "second": p2}  # dict preserves insertion order
        out = defs_from_picks(text, picks)

        assert [x.acronym for x in out] == ["PTO", "POM"]
        assert [x.definition_confidence for x in out] == [pytest.approx(0.9), pytest.approx(0.6)]


class TestDefsFromPicksIntegration:
    def test_end_to_end_pto_strips_trailing_punct_from_acr_surface(self):
        text = "Please turn over (PTO)."
        long = "Please turn over"

        a0 = text.index("PTO")
        a1 = a0 + len("PTO).")  # includes trailing ').' in the span
        d0, d1 = _span(text, long)

        picks = {
            "pto": InTextPick(
                definition=long,
                acr_span=(a0, a1),
                def_span=(d0, d1),
                definition_confidence=0.87,
                original_definition=long,
            )
        }

        out = defs_from_picks(text, picks)
        assert len(out) == 1
        assert out[0].acronym == "PTO"

    def test_end_to_end_pto_forward_form(self):
        """
        Real tighten_label_by_acronym is used. Classic forward form:
        'Please turn over (PTO)' should tighten to the full phrase.
        """
        text = "Please turn over (PTO) before proceeding."
        long = "Please turn over"

        a0, a1 = _span(text, "PTO")
        d0, d1 = _span(text, long)

        picks = {
            "pto": InTextPick(
                definition=long,
                acr_span=(a0, a1),
                def_span=(d0, d1),
                definition_confidence=0.87,
                original_definition=long,
                route="all_occ_scan_parenthetical",
            )
        }

        out = defs_from_picks(text, picks)
        assert len(out) == 1
        item = out[0]

        # Uppercasing, spans, and pass-through via tighten
        assert item.acronym == "PTO"
        assert item.definition == "Please turn over"
        assert item.original_definition == long
        assert (item.acr_start, item.acr_end) == (a0, a1)
        assert (item.def_start, item.def_end) == (d0, d1)
        assert item.source == "all_occ_scan_parenthetical"
        assert item.definition_confidence == pytest.approx(0.87)

    def test_end_to_end_gpu_mixed_case_surface(self):
        """
        Mixed-case acronym in text must be uppercased, and the tightened
        definition should remain the canonical phrase.
        """
        text = "Modern cards include a Gpu for parallel workloads."
        acr_surface = "Gpu"
        long = "Graphics Processing Unit"

        a0, a1 = _span(text, acr_surface)
        # definition span can be a synthetic one for integration coverage
        d0, d1 = (0, len(long))

        picks = {
            "gpu": InTextPick(
                definition=long,
                acr_span=(a0, a1),
                def_span=(d0, d1),
                definition_confidence=0.73,
                original_definition=long,
            )
        }

        out = defs_from_picks(text, picks)
        assert len(out) == 1
        item = out[0]

        assert item.acronym == "GPU"
        assert item.definition == "Graphics Processing Unit"
        assert item.original_definition == long
        assert item.definition_confidence == pytest.approx(0.73)

    def test_multiple_picks_stable_order_pdf_then_rom(self):
        """
        Multiple entries should preserve insertion order and each should be
        tightened correctly using the real implementation.
        """
        text = "Export as PDF, then check the ROM section."
        long1 = "Portable Document Format"
        long2 = "Read Only Memory"

        a10, a11 = _span(text, "PDF")
        a20, a21 = _span(text, "ROM")

        # synthetic definition spans (not necessarily from the same text)
        picks = {
            "pdf": InTextPick(
                definition=long1,
                acr_span=(a10, a11),
                def_span=(0, len(long1)),
                definition_confidence=0.95,
                original_definition=long1,
            ),
            "rom": InTextPick(
                definition=long2,
                acr_span=(a20, a21),
                def_span=(0, len(long2)),
                definition_confidence=0.66,
                original_definition=long2,
            ),
        }

        out = defs_from_picks(text, picks)
        assert [x.acronym for x in out] == ["PDF", "ROM"]
        assert [x.definition for x in out] == [
            "Portable Document Format",
            "Read Only Memory",
        ]
        assert [x.definition_confidence for x in out] == [
            pytest.approx(0.95),
            pytest.approx(0.66),
        ]


class TestMeaningKey:
    def test_acronym_is_uppercased(self):
        assert _meaning_key("Pdf", "Portable Document Format")[0] == "PDF"

    def test_stands_for_is_normalized(self):
        assert _meaning_key("pdf", "PDF stands for Portable Document Format")[1] == "portable document format"

    def test_trailing_proper_noun_chunk(self):
        key = _meaning_key("BIC", "The British-Irish Council")
        assert key == ("BIC", "british-irish council")

    def test_leading_connectors_are_removed_then_lowered(self):
        key = _meaning_key("PDF", "And, which the Portable Document Format")
        assert key == ("PDF", "portable document format")

    def test_acronym_and_label_are_independent(self):
        # Intentional mismatch: function does not relate acronym to label
        key = _meaning_key("GPU", "Portable Document Format")
        assert key == ("GPU", "portable document format")


def _ed(acr: str, defn: str, *, a0=0, a1=3, d0=10, d1=20, conf=0.9) -> ExtractedDefinition:
    return ExtractedDefinition(
        acronym=acr,
        definition=defn,
        source="all_occ_scan_parenthetical",
        definition_confidence=conf,
        acr_start=a0,
        acr_end=a1,
        def_start=d0,
        def_end=d1,
        original_definition=defn,
    )


class TestDedupeDefsUnit:
    def test_empty(self):
        assert dedupe_defs([]) == []

    def test_keeps_first_duplicate_and_preserves_order_no_patch(self):
        # Same meaning after real normalization
        d1 = _ed("PDF", "Portable Document Format")
        d2 = _ed("Pdf", "And, which the Portable Document Format")  # dup of d1
        d3 = _ed("GPU", "Gamma three")  # distinct

        out = dedupe_defs([d1, d2, d3])
        assert [o.definition for o in out] == ["Portable Document Format", "Gamma three"]
        assert out[0] is d1  # keep-first

    def test_does_not_modify_definition_text_no_patch(self):
        d1 = _ed("ABC", "Raw Def — Keep As-Is")
        d2 = _ed("abc", "And, which Raw Def — Keep As-Is")  # dup of d1 after tighten

        out = dedupe_defs([d1, d2])
        assert len(out) == 1
        assert out[0] is d1
        assert out[0].definition == "Raw Def — Keep As-Is"


class TestDedupeDefsIntegration:
    def test_dedupes_by_normalized_key_case_and_connectors(self):
        # These two are the "same" meaning after tighten_label/_meaning_key normalization
        d_pdf_1 = _ed("Pdf", "Portable Document Format")
        d_pdf_2 = _ed("PDF", "And, which the Portable Document Format")

        # Distinct meaning
        d_gpu = _ed("GPU", "Graphics Processing Unit")

        inp = [d_pdf_1, d_pdf_2, d_gpu]
        out = dedupe_defs(inp)

        # Keeps first occurrence of the PDF meaning, and keeps GPU
        assert [x.acronym for x in out] == ["Pdf", "GPU"]  # original casing preserved in objects
        assert [x.definition for x in out] == [
            "Portable Document Format",
            "Graphics Processing Unit",
        ]
        # Ensure dedupe kept the exact first object instance
        assert out[0] is d_pdf_1

    def test_mismatched_label_forms_dedupe_with_same_acronym(self):
        a = _ed("PDF", "Portable Document Format")
        b = _ed("pdf", "And, which the Portable Document Format")  # proven normalisation form
        out = dedupe_defs([a, b])
        assert len(out) == 1
        assert out[0] is a

    def test_order_preserved_across_multiple_groups(self):
        d1 = _ed("ROM", "Read Only Memory")
        d2 = _ed("rom", "The Read Only Memory")  # dup of d1
        d3 = _ed("HTTP", "Hypertext Transfer Protocol")
        d4 = _ed("http", "And, which Hypertext Transfer Protocol")  # dup of d3
        d5 = _ed("SQL", "Structured Query Language")

        out = dedupe_defs([d1, d2, d3, d4, d5])
        # First of each group kept, stable order overall
        assert [x.acronym for x in out] == ["ROM", "HTTP", "SQL"]
        assert [x.definition for x in out] == [
            "Read Only Memory",
            "Hypertext Transfer Protocol",
            "Structured Query Language",
        ]
