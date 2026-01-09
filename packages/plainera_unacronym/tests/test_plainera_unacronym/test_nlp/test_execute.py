import pytest
from types import SimpleNamespace as NS

import plainera_unacronym.nlp.extraction.engine.detect_flow as mod
from plainera_unacronym.nlp.common.types import DetectorResult, Occurrence, DetectorConfig, FirstOccurrence, InTextPick
from plainera_unacronym.nlp.execute import detect_and_extract
from plainera_unacronym.nlp.extraction import ExtractionConfig


def _first_occ(text: str, acr: str, start: int, confidence: float) -> FirstOccurrence:
    return FirstOccurrence(acronym=acr, start_offset=start, end_offset=start + len(acr), confidence=confidence)


def _ed(acr: str, d: str, a0: int = 0, a1: int = 0, d0: int = 0, d1: int = 0, conf: float = 0.95, src="in_text", orig=None):
    return mod.ExtractedDefinition(
        acronym=acr,
        definition=d,
        source=src,
        confidence=conf,
        acr_start=a0,
        acr_end=a1,
        def_start=d0,
        def_end=d1,
        original_definition=orig or d,
    )


def _cfgs():
    return (
        DetectorConfig(),  # default is fine
        ExtractionConfig(
            inline_cues=(r"short\s+for", r"stands?\s+for"),
            max_phrase_chars=200,
            enabled_parenthetical=True,
            enabled_inline=True,
            conf_parenthetical=0.95,
            conf_inline=0.80,
        ),
    )


class TestDetectAndExtractUnit:
    def test_strategy_anchored_plus_harvest_when_nothing_missing(self, monkeypatch):
        text = "Portable Document Format (PDF)."

        det_cfg, ext_cfg = _cfgs()

        # Fake Detector.detect -> has one FO and one occurrence
        class FakeDetector:
            cfg = det_cfg
            def __init__(self, config=None): pass
            def detect(self, t):
                return DetectorResult(
                    occurrences=[Occurrence(acronym="PDF", start_offset=28, end_offset=31, confidence=0.5, context_window=(0,32))],
                    unique_acronyms={"PDF": _first_occ(text, "PDF", 28, 0.5)},
                )

        monkeypatch.setattr(mod, "Detector", FakeDetector)

        # Anchored picks: found
        anchored_pick = InTextPick(
            definition="Portable Document Format",
            acr_span=(28, 31),
            def_span=(0, 24),
            confidence=0.98,
            original_definition="Portable Document Format",
        )
        monkeypatch.setattr(mod, "extract_near_firsts", lambda *a, **k: {"PDF": anchored_pick})

        # defs_from_picks returns one ED
        monkeypatch.setattr(mod, "defs_from_picks", lambda _text, picks: [_ed("PDF", "Portable Document Format")])

        # harvest returns nothing extra
        monkeypatch.setattr(mod, "harvest_defs_all", lambda *_: [])

        # dedupe returns what it gets
        monkeypatch.setattr(mod, "dedupe_defs", lambda defs: defs)

        # build_senses: single sense per acronym
        monkeypatch.setattr(mod, "build_senses", lambda defs: {"PDF": [NS(sense_id="PDF::Portable Document Format")]})

        # disambiguate_occurrences: one resolution using that sense
        Res = NS  # simple container
        monkeypatch.setattr(
            mod,
            "disambiguate_occurrences",
            lambda **kw: [NS(chosen_sense_id="PDF::Portable Document Format", occurrence=NS(acronym="PDF"))],
        )

        det_res, extr = detect_and_extract(text, det_cfg=det_cfg, ext_cfg=ext_cfg)

        assert extr.strategy == "anchored+harvest"
        assert extr.missing_keys == ()
        assert 0 < extr.coverage <= 1.0
        assert "PDF" in extr.senses_by_acronym
        assert extr.sense_index["PDF::Portable Document Format"].sense_id == "PDF::Portable Document Format"
        assert not extr.ambiguous_keys
        assert all(r.chosen_sense_id for r in extr.resolutions)

    def test_strategy_falls_back_to_global_when_missing(self, monkeypatch):
        text = "Only one acronym appears: ABC."

        det_cfg, ext_cfg = _cfgs()

        class FakeDetector:
            def __init__(self, config=None): pass

            def detect(self, _):
                return DetectorResult(
                    occurrences=[Occurrence(acronym="ABC", start_offset=29, end_offset=32, confidence=0.5,
                                            context_window=(0, 32))],
                    unique_acronyms={"ABC": _first_occ(text, "ABC", 29, 0.5)},
                )

        monkeypatch.setattr("plainera_unacronym.nlp.extraction.engine.detect_flow.Detector", FakeDetector)
        monkeypatch.setattr("plainera_unacronym.nlp.extraction.engine.detect_flow.extract_near_firsts",
                            lambda *a, **k: {"ABC": None})
        monkeypatch.setattr("plainera_unacronym.nlp.extraction.engine.detect_flow.defs_from_picks", lambda *_: [])
        monkeypatch.setattr("plainera_unacronym.nlp.extraction.engine.detect_flow.harvest_defs_all", lambda *_: [])

        # supply a global def via the pipeline
        monkeypatch.setattr(
            "plainera_unacronym.nlp.extraction.engine.detect_flow.extract_pipeline_iter",
            lambda text, detcfg, extcfg, plan=None: [_ed("ABC", "Alpha Beta Core", conf=0.90)],
        )
        monkeypatch.setattr("plainera_unacronym.nlp.extraction.engine.detect_flow.dedupe_defs", lambda defs: defs)
        monkeypatch.setattr("plainera_unacronym.nlp.extraction.engine.detect_flow.build_senses",
                            lambda defs: {"ABC": [NS(sense_id="s-abc")]})
        monkeypatch.setattr("plainera_unacronym.nlp.extraction.engine.detect_flow.disambiguate_occurrences",
                            lambda **kw: [NS(chosen_sense_id=None)])

        det_res, extr = detect_and_extract(text, det_cfg=det_cfg, ext_cfg=ext_cfg)
        assert extr.strategy in ("anchored+harvest+global", "anchored+harvest+global-pipeline")
        assert any(d.definition == "Alpha Beta Core" for d in extr.definitions)


def _cfg_integrated(require_two_words=True, max_chars=200):
    return (
        DetectorConfig(),
        ExtractionConfig(
            inline_cues=(r"short\s+for", r"stands?\s+for", r"is\s+(?:an\s+)?acronym\s+for"),
            max_phrase_chars=max_chars,
            enabled_parenthetical=True,
            enabled_inline=True,
            conf_parenthetical=0.95,
            conf_inline=0.80,
            require_two_words=require_two_words,
        ),
    )



class TestDetectAndExtractIntegration:
    def test_mixed_forward_reverse_inline_and_confidence(self):
        text = (
            "We invest in Research and Development (R&D) to innovate.\n"
            "The CFO said C/A (Cost per Acquisition) has fallen this quarter.\n"
            "PTO stands for Please Turn Over on print jobs.\n"
            "Finally, AM, short for amplitude modulation, is a legacy technique.\n"
            "Portable Document Format (PDF) dominates documents; elsewhere PDF (Portable Document Format) appears."
        )

        det_cfg, ext_cfg = _cfg_integrated()

        det_res, extr = detect_and_extract(text, det_cfg=det_cfg, ext_cfg=ext_cfg)

        by = {}
        for d in extr.definitions:
            by.setdefault(d.acronym, []).append(d)

        # R&D forward
        assert "R&D" in by
        assert any("Research and Development" in e.definition for e in by["R&D"])
        assert all(0 < e.confidence <= 0.99 for e in by["R&D"])

        # C/A reverse
        assert "C/A" in by
        assert any(e.definition == "Cost per Acquisition" for e in by["C/A"])

        # PTO inline
        assert "PTO" in by
        assert any("Please Turn Over" in e.definition for e in by["PTO"])

        # AM inline
        assert "AM" in by
        assert any("amplitude modulation" in e.definition.lower() for e in by["AM"])

        # PDF appears
        assert "PDF" in by
        assert any("Portable Document Format" in e.definition for e in by["PDF"])


class TestDetectAndExtractIntegrationEdgeCases:
    def test_forward_parenthetical_does_not_span_newlines_and_keeps_pto(self):
        # PTO line must not be swallowed by the forward (PDF) match
        text = (
            "PTO stands for Please Turn Over on print jobs.\n"
            "Portable Document Format (PDF) dominates documents."
        )
        det_cfg, ext_cfg = _cfg_integrated()
        det_res, extr = detect_and_extract(text, det_cfg=det_cfg, ext_cfg=ext_cfg)

        by = {}
        for d in extr.definitions:
            by.setdefault(d.acronym, []).append(d)

        assert "PTO" in by
        assert any("Please Turn Over" in e.definition for e in by["PTO"])

        assert "PDF" in by
        assert any("Portable Document Format" in e.definition for e in by["PDF"])

    def test_inline_long_tail_is_gated_by_max_phrase_chars(self):
        # With a strict max, PTO should be dropped; with relaxed max, it should appear
        base = (
            "In printing, PTO stands for "
            "a very, very long descriptive phrase that should be trimmed or rejected entirely "
            "depending on configuration and normalisation steps. "
            "Portable Document Format (PDF) is common."
        )
        # Strict
        det_cfg, ext_cfg_strict = _cfg_integrated(require_two_words=True, max_chars=20)
        det, extr, reports, trace = detect_and_extract(
            base, det_cfg=det_cfg, ext_cfg=ext_cfg_strict,
            return_reports=True, trace=True, trace_filter=r"^(PTO|PF)$"
        )

        for r in reports:
            print(f"{r.name:22} :: {r.info}")

        from pprint import pprint
        pprint(trace)

        assert not any(d.acronym == "PTO" for d in extr.definitions)
        assert any(d.acronym == "PDF" for d in extr.definitions)

        # Relaxed
        det_cfg, ext_cfg_relaxed = _cfg_integrated(require_two_words=True, max_chars=160)
        det_res_r, extr_r = detect_and_extract(base, det_cfg=det_cfg, ext_cfg=ext_cfg_relaxed)

        by_r = {}
        for d in extr_r.definitions:
            by_r.setdefault(d.acronym, []).append(d)

        assert "PTO" in by_r
        assert any(e.definition.strip() for e in by_r["PTO"])
        assert any(0 < e.confidence <= 0.99 for e in by_r["PTO"])

    def test_numeric_leading_token_is_preserved_in_parenthetical_after(self):
        # Ensure numeric-leading tokens (e.g., 3M) are kept in the definition window
        text = "PF (3M Portable format) is a special case in this doc."
        det_cfg, ext_cfg = _cfg_integrated()
        det_res, extr = detect_and_extract(text, det_cfg=det_cfg, ext_cfg=ext_cfg)
        by = {}
        for d in extr.definitions:
            by.setdefault(d.acronym, []).append(d)

        assert "PF" in by
        assert any(e.definition.startswith("3M ") for e in by["PF"])

    def test_special_char_acronym_and_bridges(self):
        # Exercise slash (& keeps) and dash/ampersand cases
        text = (
            "We track C/A (Cost per Acquisition) closely. "
            "Research & Development (R&D) invests heavily."
        )
        det_cfg, ext_cfg = _cfg_integrated()
        det_res, extr = detect_and_extract(text, det_cfg=det_cfg, ext_cfg=ext_cfg)

        by = {}
        for d in extr.definitions:
            by.setdefault(d.acronym, []).append(d)

        assert "C/A" in by
        assert any(e.definition == "Cost per Acquisition" for e in by["C/A"])

        assert "R&D" in by
        assert any("Research & Development" in e.definition for e in by["R&D"])

    def test_ambiguous_acronym_builds_multiple_senses(self):
        # EMA appears with two meanings; result should have ambiguous senses for EMA
        text = (
            "EMA stands for European Medicines Agency in the EU context. "
            "On charts, EMA (Exponential Moving Average) is a common indicator."
        )
        det_cfg, ext_cfg = _cfg_integrated()
        det_res, extr = detect_and_extract(text, det_cfg=det_cfg, ext_cfg=ext_cfg)

        # senses_by_acronym and ambiguous_keys should reflect two senses for EMA
        senses = extr.senses_by_acronym.get("EMA", [])
        assert len(senses) >= 2, f"Expected multiple senses for EMA, got {senses}"
        assert "EMA" in extr.ambiguous_keys

        # And definitions should include both
        defs_for_ema = [d.definition for d in extr.definitions if d.acronym == "EMA"]
        joined = " || ".join(defs_for_ema).lower()
        assert "european medicines agency" in joined
        assert "exponential moving average" in joined

    def test_nearest_pick_prefers_definition_near_first_occurrence(self):
        # Two candidate long-forms for the same acronym; ensure the one closest to the FO wins
        text = (
            "Portable Document Format (PDF) is ubiquitous. "     # <-- near FO
            "Later we see some detour text and a PDF (Pretty Darn Fast) joke."
        )
        det_cfg, ext_cfg = _cfg_integrated()
        det_res, extr = detect_and_extract(text, det_cfg=det_cfg, ext_cfg=ext_cfg)

        # The chosen pick for PDF (by nearest) should be the first proper definition
        pick = extr.picks.get("PDF")
        assert pick is not None
        assert "Portable Document Format" in pick.definition
