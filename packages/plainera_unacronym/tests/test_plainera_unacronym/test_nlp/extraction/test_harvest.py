from types import SimpleNamespace
import pytest

from plainera_unacronym.nlp.extraction.harvest import harvest_defs_all


class Cfg:
    def __init__(self, window_chars=320):
        self.window_chars = window_chars


class Occ:
    def __init__(self, acronym: str, start: int, end: int):
        self.acronym = acronym
        self.start_offset = start
        self.end_offset = end


def _patch(monkeypatch, func, **repl):
    g = func.__globals__
    for name, impl in repl.items():
        monkeypatch.setitem(g, name, impl)


class TestHarvestDefsAllUnit:
    def test_before_path_maps_absolute_spans_and_uppercases_for_tighten(self, monkeypatch):
        # Text:           0123456789012345678901234567890123456789
        text =           "AAA Portable Document Format (PDF) ZZZ"
        acr0 = text.index("PDF")
        acr1 = acr0 + 3
        occs = [Occ("Pdf", acr0, acr1)]  # original case retained in output

        # pre will be text[L:R] slice up to rel_a1+1; we simulate one before-match:
        # def is exactly "Portable Document Format" spanning indices:
        d0 = text.index("Portable")
        d1 = d0 + len("Portable Document Format")

        fake_before = [
            SimpleNamespace(def_start=d0, def_end=d1, definition="Portable Document Format")
        ]
        # The after finder should not be called for this test; return empty
        _patch(
            monkeypatch, harvest_defs_all,
            find_longform_before_acr=lambda snippet, acr, cfg: [
                SimpleNamespace(
                    def_start=d0,  # note: in harvest, we expect these to be relative to snippet start L
                    def_end=d1,
                    definition="Portable Document Format",
                )
            ] if snippet in (text[:acr1+1], text[:acr1]) else [],
            find_longform_after_acr=lambda right, cfg, acr=None: [],
            tighten_label_by_acronym=lambda raw, acr_up: f"TIGHT[{raw}|{acr_up}]",
        )

        # To make def_start/def_end values relative to snippet L, force L=0 with large window
        cfg = Cfg(window_chars=len(text))
        out = harvest_defs_all(text, occs, cfg)
        assert len(out) == 1
        item = out[0]

        # Acronym preserved from occ as-is
        assert item.acronym == "Pdf"
        # Tightener received the UPPER acronym
        assert item.definition == "TIGHT[Portable Document Format|PDF]"
        # def_start/def_end are absolute and slice the original text exactly
        assert text[item.def_start:item.def_end] == "Portable Document Format"
        # acr span mapped from occ
        assert (item.acr_start, item.acr_end) == (acr0, acr1)
        assert item.source == "in_text"
        assert item.original_definition == "Portable Document Format"
        assert item.confidence == pytest.approx(0.95)

    def test_after_path_maps_relative_right_offset(self, monkeypatch):
        text = "See GPU (Graphics Processing Unit) for details"
        acr0 = text.index("GPU")
        acr1 = acr0 + 3
        occs = [Occ("GPU", acr0, acr1)]

        # right = snippet[rel_a1:], so m.def_start/def_end are relative to 'right'
        # make them point exactly to "Graphics Processing Unit"
        inner = "Graphics Processing Unit"
        inner_start = text.index(inner)
        inner_end = inner_start + len(inner)

        def fake_after(right, cfg, acr=None):
            # right is text starting at acr1 when L=0 (big window)
            # so def_start is inner_start - acr1
            return [SimpleNamespace(
                def_start=inner_start - acr1,
                def_end=inner_end - acr1,
                definition=inner,
            )]

        cfg = Cfg(window_chars=len(text))
        _patch(
            monkeypatch, harvest_defs_all,
            find_longform_before_acr=lambda *a, **k: [],
            find_longform_after_acr=fake_after,
            tighten_label_by_acronym=lambda raw, acr_up: raw,
        )

        out = harvest_defs_all(text, occs, cfg)
        assert len(out) == 1
        item = out[0]
        assert item.acronym == "GPU"
        assert item.definition == "Graphics Processing Unit"
        assert text[item.def_start:item.def_end] == inner
        assert (item.acr_start, item.acr_end) == (acr0, acr1)

    def test_window_clamps_and_multiple_occs_accumulate(self, monkeypatch):
        # Occ #1 near start; Occ #2 near end; ensure window clamps 0..len(text)
        text = "PDF (Portable Document Format) ... lead-in ... GPU (Graphics Processing Unit)"
        pdf0 = text.index("PDF")
        pdf1 = pdf0 + 3
        gpu0 = text.index("GPU")
        gpu1 = gpu0 + 3

        cfg = Cfg(window_chars=10)  # small, forces clamps

        def fake_before(snippet, acr, cfg_):
            if acr == "PDF":
                # pre covers "(Portable Document Format)" in this snippet
                start = snippet.index("Portable")
                end = start + len("Portable Document Format")
                return [SimpleNamespace(def_start=start, def_end=end, definition="Portable Document Format")]
            return []

        def fake_after(snippet, cfg_, acr=None):
            if acr == "GPU":
                start = snippet.index("Graphics")
                end = start + len("Graphics Processing Unit")
                return [SimpleNamespace(def_start=start, def_end=end, definition="Graphics Processing Unit")]
            return []

        _patch(
            monkeypatch, harvest_defs_all,
            find_longform_before_acr=lambda *a, **k: [],
            find_longform_after_acr=fake_after,
            tighten_label_by_acronym=lambda raw, acr_up: raw,
        )

        occs = [Occ("PDF", pdf0, pdf1), Occ("GPU", gpu0, gpu1)]
        out = harvest_defs_all(text, occs, cfg)
        # Both definitions harvested
        assert [o.acronym for o in out] == ["PDF", "GPU"]
        assert [o.definition for o in out] == [
            "Portable Document Format",
            "Graphics Processing Unit",
        ]


class TestHarvestDefsAllIntegration:
    def test_before_and_after_patterns_end_to_end(self):
        text = (
            "We refer to the Portable Document Format (PDF) throughout. "
            "Modern cards include a GPU (Graphics Processing Unit) for parallel workloads."
        )
        pdf0 = text.index("PDF")
        pdf1 = pdf0 + 3
        gpu0 = text.index("GPU")
        gpu1 = gpu0 + 3

        cfg = Cfg(window_chars=80)
        occs = [Occ("PDF", pdf0, pdf1), Occ("GPU", gpu0, gpu1)]
        out = harvest_defs_all(text, occs, cfg)

        # We expect two results, one from "before", one from "after"
        assert len(out) == 2

        # Map by acronym to make assertions clearer
        by_acr = {o.acronym: o for o in out}

        pdf = by_acr["PDF"]
        assert pdf.source == "in_text"
        assert pdf.original_definition == "Portable Document Format"
        # tightened label should still be the same canonical phrase
        assert pdf.definition == "Portable Document Format"
        assert text[pdf.def_start:pdf.def_end] == "Portable Document Format"
        assert (pdf.acr_start, pdf.acr_end) == (pdf0, pdf1)
        assert pdf.confidence == pytest.approx(0.95)

        gpu = by_acr["GPU"]
        assert gpu.original_definition == "Graphics Processing Unit"
        assert gpu.definition == "Graphics Processing Unit"
        assert text[gpu.def_start:gpu.def_end] == "Graphics Processing Unit"
        assert (gpu.acr_start, gpu.acr_end) == (gpu0, gpu1)

    def test_window_chars_limits_search_region(self):
        # Window too small to include the definition — expect no results
        text = "Portable Document Format (PDF) appears here."
        pdf0 = text.index("PDF")
        pdf1 = pdf0 + 3

        # Window that doesn't reach the words before "PDF"
        cfg = Cfg(window_chars=1)
        out = harvest_defs_all(text, [Occ("PDF", pdf0, pdf1)], cfg)
        assert out == []

        # Large enough window finds it
        cfg2 = Cfg(window_chars=50)
        out2 = harvest_defs_all(text, [Occ("PDF", pdf0, pdf1)], cfg2)
        assert len(out2) == 1
        assert out2[0].definition == "Portable Document Format"
