from types import SimpleNamespace

import pytest
from plainera_unacronym.nlp.extraction.strategies.harvest import extract_defs_all_occurrences


class Cfg:
    def __init__(self, window_chars=320):
        self.window_chars = window_chars


class Occ:
    def __init__(self, acronym: str, start: int, end: int):
        self.acronym = acronym
        self.start_offset = start
        self.end_offset = end


class TestExtractDefsAllOccurrencesUnit:
    def test_before_path_maps_absolute_spans_and_uppercases_for_tighten(self, _patch):
        # Text:           0123456789012345678901234567890123456789
        text = "AAA Portable Document Format (PDF) ZZZ"
        acr0 = text.index("PDF")
        acr1 = acr0 + 3
        occs = [Occ("Pdf", acr0, acr1)]  # original case retained in output

        # pre will be text[L:R] slice up to rel_a1+1; we simulate one before-match:
        # def is exactly "Portable Document Format" spanning indices:
        d0 = text.index("Portable")
        d1 = d0 + len("Portable Document Format")

        # The after finder should not be called for this test; return empty
        _patch(
            extract_defs_all_occurrences,
            find_parenthetical_longform_before_acr=lambda snippet, acr, cfg: [
                SimpleNamespace(
                    def_start=d0,  # note: in harvest, we expect these to be relative to snippet start L
                    def_end=d1,
                    definition="Portable Document Format",
                )
            ] if snippet in (text[:acr1 + 1], text[:acr1]) else [],
            find_parenthetical_longform_after_acr=lambda right, cfg, acr=None: [],
            tighten_label_by_acronym=lambda raw, acr_up: f"TIGHT[{raw}|{acr_up}]",
        )

        # To make def_start/def_end values relative to snippet L, force L=0 with large window
        cfg = Cfg(window_chars=len(text))
        out = extract_defs_all_occurrences(text, occs, cfg)
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
        assert item.source == "all_occ_scan_parenthetical"
        assert item.original_definition == "Portable Document Format"
        assert item.definition_confidence == pytest.approx(0.95)

    def test_after_path_maps_relative_right_offset(self, _patch):
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
            extract_defs_all_occurrences,
            find_parenthetical_longform_before_acr=lambda *a, **k: [],
            find_parenthetical_longform_after_acr=fake_after,
            tighten_label_by_acronym=lambda raw, acr_up: raw,
        )

        out = extract_defs_all_occurrences(text, occs, cfg)
        assert len(out) == 1
        item = out[0]
        assert item.acronym == "GPU"
        assert item.definition == "Graphics Processing Unit"
        assert text[item.def_start:item.def_end] == inner
        assert (item.acr_start, item.acr_end) == (acr0, acr1)

    def test_window_clamps_and_multiple_occs_accumulate(self, _patch):
        text = "PDF (Portable Document Format) ... lead-in ... GPU (Graphics Processing Unit)"
        pdf0 = text.index("PDF")
        pdf1 = pdf0 + 3
        gpu0 = text.index("GPU")
        gpu1 = gpu0 + 3

        cfg = Cfg(window_chars=len(text))  # big window: include both phrases

        def fake_before(snippet, acr, cfg):
            if acr != "PDF":
                return []
            phrase = "Portable Document Format"
            if phrase in snippet:
                s = snippet.index(phrase)
                e = s + len(phrase)
                return [SimpleNamespace(def_start=s, def_end=e, definition=phrase)]
            return []

        def fake_after(snippet, cfg, acr=None, **_):
            mapping = {
                "PDF": "Portable Document Format",
                "GPU": "Graphics Processing Unit",
            }
            phrase = mapping.get(acr)
            if not phrase:
                return []
            if phrase in snippet:
                s = snippet.index(phrase)
                e = s + len(phrase)
                return [SimpleNamespace(def_start=s, def_end=e, definition=phrase)]
            return []

        _patch(
            extract_defs_all_occurrences,
            find_parenthetical_longform_before_acr=fake_before,
            find_parenthetical_longform_after_acr=fake_after,
            tighten_label_by_acronym=lambda raw, up: raw,
        )

        occs = [Occ("PDF", pdf0, pdf1), Occ("GPU", gpu0, gpu1)]
        out = extract_defs_all_occurrences(text, occs, cfg)

        assert [o.acronym for o in out] == ["PDF", "GPU"]
        assert [o.definition for o in out] == [
            "Portable Document Format",
            "Graphics Processing Unit",
        ]

    def test_small_window_finds_nothing(self, _patch):
        text = "PDF (Portable Document Format) ... GPU (Graphics Processing Unit)"
        pdf0 = text.index("PDF")
        pdf1 = pdf0 + 3
        gpu0 = text.index("GPU")
        gpu1 = gpu0 + 3
        cfg = Cfg(window_chars=3)

        _patch(
            extract_defs_all_occurrences,
            find_parenthetical_longform_before_acr=lambda *a, **k: [],
            find_parenthetical_longform_after_acr=lambda *a, **k: [],
            tighten_label_by_acronym=lambda raw, up: raw,
        )

        out = extract_defs_all_occurrences(text, [Occ("PDF", pdf0, pdf1), Occ("GPU", gpu0, gpu1)], cfg)
        assert out == []

    def test_appends_results_from_both_before_and_after_for_same_occ(self, _patch):
        text = "Portable Document Format (PDF) is common."
        acr0 = text.index("PDF")
        acr1 = acr0 + 3
        occs = [Occ("PDF", acr0, acr1)]
        cfg = Cfg(window_chars=len(text))

        phrase = "Portable Document Format"
        d0 = text.index(phrase)
        d1 = d0 + len(phrase)

        before_match = SimpleNamespace(def_start=d0, def_end=d1, definition=phrase)
        # after matcher reports spans relative to `right = snippet[rel_a1:]`
        after_match = SimpleNamespace(def_start=(d0 - acr1), def_end=(d1 - acr1), definition=phrase)

        _patch(
            extract_defs_all_occurrences,
            find_parenthetical_longform_before_acr=lambda pre, acr, cfg: [before_match],
            find_parenthetical_longform_after_acr=lambda right, cfg, acr=None: [after_match],
            tighten_label_by_acronym=lambda raw, up: raw,
        )

        out = extract_defs_all_occurrences(text, occs, cfg)
        assert len(out) == 2
        # both should map to the same absolute span and same original slice
        for item in out:
            assert (item.acr_start, item.acr_end) == (acr0, acr1)
            assert text[item.def_start:item.def_end] == phrase
            assert item.original_definition == phrase
            assert item.definition == phrase
            assert item.source == "all_occ_scan_parenthetical"
            assert item.definition_confidence == pytest.approx(0.95)

    def test_window_clamp_L_nonzero_maps_before_match_relative_spans_to_absolute(self, _patch):
        # Build text with a prefix so L will be > 0 when window clamps.
        prefix = "0123456789"  # len=10
        body = "Portable Document Format (PDF) is common."
        text = prefix + body

        acr0 = text.index("PDF")
        acr1 = acr0 + 3

        # Choose a window smaller than acr0 so L>0, but large enough to still include the definition.
        win = 30
        cfg = Cfg(window_chars=win)
        occs = [Occ("PDF", acr0, acr1)]

        # What harvest will compute:
        L = max(0, acr0 - win)
        R = min(len(text), acr1 + win)
        snippet = text[L:R]
        _rel_a1 = acr1 - L

        # `pre` is snippet up to rel_a1+1, and BEFORE matcher spans are relative to snippet start (L).
        phrase = "Portable Document Format"
        abs_d0 = text.index(phrase)
        abs_d1 = abs_d0 + len(phrase)
        rel_d0 = abs_d0 - L
        rel_d1 = abs_d1 - L

        def fake_before(pre, acr, cfg):
            # sanity: definition must be inside the clamped snippet region
            assert phrase in snippet
            return [SimpleNamespace(def_start=rel_d0, def_end=rel_d1, definition=phrase)]

        _patch(
            extract_defs_all_occurrences,
            find_parenthetical_longform_before_acr=fake_before,
            find_parenthetical_longform_after_acr=lambda *a, **k: [],
            tighten_label_by_acronym=lambda raw, up: raw,
        )

        out = extract_defs_all_occurrences(text, occs, cfg)
        assert len(out) == 1
        item = out[0]

        # This is the whole point: absolute mapping must be correct when L != 0
        assert (item.def_start, item.def_end) == (abs_d0, abs_d1)
        assert text[item.def_start:item.def_end] == phrase
        assert item.original_definition == phrase
        assert (item.acr_start, item.acr_end) == (acr0, acr1)


class TestExtractDefsAllOccurrencesIntegration:
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
        out = extract_defs_all_occurrences(text, occs, cfg)

        # We expect two results, one from "before", one from "after"
        assert len(out) == 2

        # Map by acronym to make assertions clearer
        by_acr = {o.acronym: o for o in out}

        pdf = by_acr["PDF"]
        assert pdf.source == "all_occ_scan_parenthetical"
        assert pdf.original_definition == "Portable Document Format"
        # tightened label should still be the same canonical phrase
        assert pdf.definition == "Portable Document Format"
        assert text[pdf.def_start:pdf.def_end] == "Portable Document Format"
        assert (pdf.acr_start, pdf.acr_end) == (pdf0, pdf1)
        assert pdf.definition_confidence == pytest.approx(0.95)

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
        out = extract_defs_all_occurrences(text, [Occ("PDF", pdf0, pdf1)], cfg)
        assert out == []

        # Large enough window finds it
        cfg2 = Cfg(window_chars=50)
        out2 = extract_defs_all_occurrences(text, [Occ("PDF", pdf0, pdf1)], cfg2)
        assert len(out2) == 1
        assert out2[0].definition == "Portable Document Format"

    def test_after_pattern_qae_parenthetical(self):
        # Use the “after ACR” pattern: QAE (Queen's Award for Enterprise)
        text = (
            "Winners of QAE (Queen's Award for Enterprise) were announced today."
        )
        acr0 = text.index("QAE")
        acr1 = acr0 + 3  # end offset is exclusive by convention in the codebase

        cfg = Cfg(window_chars=len(text))  # ensure the window covers the whole phrase
        occs = [Occ("QAE", acr0, acr1)]

        out = extract_defs_all_occurrences(text, occs, cfg)
        assert len(out) == 1

        item = out[0]
        # Acronym comes from the occurrence as-is
        assert item.acronym == "QAE"
        assert item.source == "all_occ_scan_parenthetical"
        assert item.definition_confidence == pytest.approx(0.95)
        assert (item.acr_start, item.acr_end) == (acr0, acr1)

        # Original definition should be the normalized parenthetical content
        expected_phrase = "Queen's Award for Enterprise"
        assert item.original_definition == expected_phrase

        # Span should tightly slice that phrase from the original text
        assert text[item.def_start: item.def_end] == expected_phrase

        # Tightened label by acronym should keep the matched tokens; depending on bridges,
        # it may keep “for”. Accept either the full phrase or the pruned variant.
        assert item.definition in (
            "Queen's Award for Enterprise",
            "Queen's Award Enterprise",
        )

    def test_before_pattern_qae_parenthetical(self):
        text = "Winners of Queen's Award for Enterprise (QAE) were announced today."
        acr0 = text.index("QAE")
        acr1 = acr0 + 3

        cfg = Cfg(window_chars=len(text))  # ensure the window covers the whole phrase
        occs = [Occ("QAE", acr0, acr1)]

        out = extract_defs_all_occurrences(text, occs, cfg)
        assert len(out) == 1

        item = out[0]
        assert item.acronym == "QAE"
        assert item.source == "all_occ_scan_parenthetical"
        assert item.definition_confidence == pytest.approx(0.95)
        assert (item.acr_start, item.acr_end) == (acr0, acr1)

        expected_phrase = "Queen's Award for Enterprise"
        # Tight span over the original text:
        assert text[item.def_start:item.def_end] == expected_phrase
        # Original definition should match the parenthetical content (after normalization):
        assert item.original_definition == expected_phrase

        # Tightening may drop bridge/stop words depending on config;
        # accept either the full phrase or a pruned variant.
        assert item.definition in {
            "Queen's Award for Enterprise",
            "Queen's Award Enterprise",
        }
