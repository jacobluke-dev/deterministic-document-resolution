from plainera_unacronym.nlp.common.types import ExtractedDefinition, Span
import plainera_unacronym.nlp.extraction.core.defs as mod
import pytest
from plainera_unacronym.nlp.common.types import InTextPick  # noqa: E402
from plainera_unacronym.nlp.extraction.core.defs import _sense_key, dedupe_defs, defs_from_picks


def _span(text: str, needle: str) -> Span:
    i = text.index(needle)
    return i, i + len(needle)


class TestDefsFromPicks:
    def test_returns_empty_for_empty_input(self, monkeypatch):
        # Tighten shouldn't be called, but keep deterministic just in case
        monkeypatch.setattr(mod, "tighten_label_by_acronym", lambda *a, **k: "N/A", raising=True)
        assert defs_from_picks("", {}) == []

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
                confidence=0.91,
                original_definition=long,
            ),
            "skip": None,
        }

        out = defs_from_picks(text, picks)
        assert len(out) == 1
        item = out[0]

        assert item.source == "in_text"
        assert item.acronym == "PTO"  # uppercased
        assert item.definition == "TIGHT[Please turn over|PTO]"
        assert item.original_definition == long
        assert (item.acr_start, item.acr_end) == (a0, a1)
        assert (item.def_start, item.def_end) == (d0, d1)
        assert item.confidence == pytest.approx(0.91)

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
            confidence=0.5,
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
        assert [x.confidence for x in out] == [pytest.approx(0.9), pytest.approx(0.6)]


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
                confidence=0.87,
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
                confidence=0.87,
                original_definition=long,
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
        assert item.source == "in_text"
        assert item.confidence == pytest.approx(0.87)

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
                confidence=0.73,
                original_definition=long,
            )
        }

        out = defs_from_picks(text, picks)
        assert len(out) == 1
        item = out[0]

        assert item.acronym == "GPU"
        assert item.definition == "Graphics Processing Unit"
        assert item.original_definition == long
        assert item.confidence == pytest.approx(0.73)

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
                confidence=0.95,
                original_definition=long1,
            ),
            "rom": InTextPick(
                definition=long2,
                acr_span=(a20, a21),
                def_span=(0, len(long2)),
                confidence=0.66,
                original_definition=long2,
            ),
        }

        out = defs_from_picks(text, picks)
        assert [x.acronym for x in out] == ["PDF", "ROM"]
        assert [x.definition for x in out] == [
            "Portable Document Format",
            "Read Only Memory",
        ]
        assert [x.confidence for x in out] == [
            pytest.approx(0.95),
            pytest.approx(0.66),
        ]


class TestSenseKey:
    def test_acronym_is_uppercased(self):
        assert _sense_key("Pdf", "Portable Document Format")[0] == "PDF"

    def test_stands_for_is_normalized(self):
        assert _sense_key("pdf", "PDF stands for Portable Document Format")[1] == "portable document format"

    def test_trailing_proper_noun_chunk(self):
        key = _sense_key("BIC", "The British-Irish Council")
        assert key == ("BIC", "british-irish council")

    def test_leading_connectors_are_removed_then_lowered(self):
        key = _sense_key("PDF", "And, which the Portable Document Format")
        assert key == ("PDF", "portable document format")

    def test_acronym_and_label_are_independent(self):
        # Intentional mismatch: function does not relate acronym to label
        key = _sense_key("GPU", "Portable Document Format")
        assert key == ("GPU", "portable document format")


def _ed(acr: str, defn: str, *, a0=0, a1=3, d0=10, d1=20, conf=0.9) -> ExtractedDefinition:
    return ExtractedDefinition(
        acronym=acr,
        definition=defn,
        source="in_text",
        confidence=conf,
        acr_start=a0, acr_end=a1,
        def_start=d0, def_end=d1,
        original_definition=defn,
    )


class TestDedupeDefsUnit:
    def test_empty(self):
        assert dedupe_defs([]) == []

    def test_keeps_first_duplicate_and_preserves_order_no_patch(self):
        # Same sense after real normalization
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
        # These two are the "same" sense after tighten_label/_sense_key normalization
        d_pdf_1 = _ed("Pdf", "Portable Document Format")
        d_pdf_2 = _ed("PDF", "And, which the Portable Document Format")

        # Distinct sense
        d_gpu = _ed("GPU", "Graphics Processing Unit")

        inp = [d_pdf_1, d_pdf_2, d_gpu]
        out = dedupe_defs(inp)

        # Keeps first occurrence of the PDF sense, and keeps GPU
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
