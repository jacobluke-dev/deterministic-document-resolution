import pprint

import pytest
from types import SimpleNamespace as NS

import plainera_unacronym.nlp.extraction.engine.detect_flow as mod
from plainera_unacronym.nlp.common.types import DetectorResult, Occurrence, DetectorConfig, FirstOccurrence, InTextPick
from plainera_unacronym.nlp.execute import detect_and_extract
from plainera_unacronym.nlp.extraction import ExtractionConfig


def picked_def(extr, key: str):
    """Return extracted definition for acronym key if present, else None."""
    pick = extr.picks.get(key)
    if pick is None:
        return None
    return pick.definition


def _first_occ(text: str, acr: str, start: int, confidence: float) -> FirstOccurrence:
    return FirstOccurrence(acronym=acr, start_offset=start, end_offset=start + len(acr), confidence=confidence)


def _ed(acr: str, d: str, a0: int = 0, a1: int = 0, d0: int = 0, d1: int = 0, conf: float = 0.95, src="in_text",
        orig=None):
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
                    occurrences=[Occurrence(acronym="PDF", start_offset=28, end_offset=31, confidence=0.5,
                                            context_window=(0, 32))],
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

    def test_inline_long_tail_is_gated_by_max_phrase_chars_30(self):
        # With a strict max, PTO should be dropped; with relaxed max, it should appear
        base = (
            "In printing, PTO stands for "
            "a very, very long descriptive phrase that should be trimmed or rejected entirely "
            "depending on configuration and normalisation steps. "
            "Portable Document Format (PDF) is common."
        )
        # Strict
        det_cfg, ext_cfg_strict = _cfg_integrated(require_two_words=True, max_chars=30)
        det, extr, reports, trace = detect_and_extract(
            base, det_cfg=det_cfg, ext_cfg=ext_cfg_strict,
            return_reports=True, trace=True, trace_filter=r"^(PTO|PF)$"
        )

        for r in reports:
            print(f"{r.name:22} :: {r.info}")

        from pprint import pprint
        pprint(extr)
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

    def test_inline_long_tail_is_gated_by_max_phrase_chars_40(self):
        # With a strict max, PTO should be dropped; with relaxed max, it should appear
        base = (
            "In printing, PTO stands for "
            "a very, very long descriptive phrase that should be trimmed or rejected entirely "
            "depending on configuration and normalisation steps. "
            "Portable Document Format (PDF) is common."
        )
        # Strict
        det_cfg, ext_cfg_strict = _cfg_integrated(require_two_words=True, max_chars=40)
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
            "Portable Document Format (PDF) is ubiquitous. "  # <-- near FO
            "Later we see some detour text and a PDF (Pretty Darn Fast) joke."
        )
        det_cfg, ext_cfg = _cfg_integrated()
        det_res, extr = detect_and_extract(text, det_cfg=det_cfg, ext_cfg=ext_cfg)

        # The chosen pick for PDF (by nearest) should be the first proper definition
        pick = extr.picks.get("PDF")
        assert pick is not None
        assert "Portable Document Format" in pick.definition

    def test_tier_one_square_brackets_definition(self):
        det, extr = detect_and_extract("Portable Document Format [PDF] is common.")
        assert picked_def(extr, "PDF") in {"Portable Document Format"}

    def test_tier_one_parenthetical_quotes_around_acronym(self):
        det, extr = detect_and_extract('Portable Document Format ("PDF") is common.')
        assert picked_def(extr, "PDF") == "Portable Document Format"

    def test_tier_one_parenthetical_tail_colon(self):
        det, extr = detect_and_extract("Personal protective equipment (PPE: required on site) matters.")
        assert picked_def(extr, "PPE") == "Personal protective equipment"

    def test_tier_one_parenthetical_tail_dash(self):
        det, extr = detect_and_extract("Personal protective equipment (PPE - required on site) matters.")
        assert picked_def(extr, "PPE") == "Personal protective equipment"

    def test_tier_one_multiple_occurrences_one_definition(self):
        det, extr = detect_and_extract(
            "Portable Document Format (PDF) is common. PDF files are everywhere."
        )
        assert picked_def(extr, "PDF") == "Portable Document Format"

    def test_tier_one_digit_prefixed_acronym_parenthetical(self):
        det, extr = detect_and_extract("Third Generation Partnership Project (3GPP) publishes specs.")
        assert picked_def(extr, "3GPP") == "Third Generation Partnership Project"

    def test_tier_one_mixed_digits_acronym_parenthetical(self):
        det, extr = detect_and_extract("Hypertext Transfer Protocol 2 (HTTP2) is used.")
        assert picked_def(extr, "HTTP2") == "Hypertext Transfer Protocol 2"

    def test_tier_one_definition_before_acronym_does_not_capture_trailing_space(self):
        det, extr = detect_and_extract("Portable Document Format, (PDF) is common.")
        assert picked_def(extr, "PDF") == "Portable Document Format"

    def test_tier_one_pick_kind_is_set_for_anchored_parenthetical(self):
        det, extr = detect_and_extract("Portable Document Format (PDF) is common.")
        assert extr.picks["PDF"] is not None
        assert getattr(extr.picks["PDF"], "kind", None) in {"def_before", "def_after"}  # whatever you standardise


class TestDetectAndExtractE2E:

    def test_detect_and_extract_dash_mixed_case(self):
        # 1) Lower-case tokens should be preserved (no truncation)
        det, extr = detect_and_extract("Single sign-on (SSO) is enabled.")
        assert picked_def(extr, "SSO") in {"Single sign-on"}, extr.picks.get("SSO")

    def test_parenthetical_preserves_lowercase_hyphen_token(self):
        det, extr, reports = detect_and_extract("single sign-on (SSO) is enabled.", return_reports=True)
        pprint.pprint(reports)
        pprint.pprint(extr)
        assert picked_def(extr, "SSO") == "single sign-on", extr.picks.get("SSO")

    def test_parenthetical_all_lowercase_definition_is_allowed(self):
        det, extr = detect_and_extract("return on investment (ROI) is tracked.")
        assert picked_def(extr, "ROI") == "return on investment", extr.picks.get("ROI")

    def test_parenthetical_acronym_only_is_rejected(self):
        det, extr = detect_and_extract(
            "We discussed options and agreed on the approach (SLA) yesterday."
        )
        assert extr.picks.get("SLA") is None, extr.picks.get("SLA")

    def test_parenthetical_proper_noun_definition_is_extracted(self):
        det, extr = detect_and_extract("National Health Service (NHS) guidelines apply.")
        assert picked_def(extr, "NHS") == "National Health Service", extr.picks.get("NHS")

    def test_tier_one_e2e(self):
        det, extr = detect_and_extract("Our encryption is end-to-end (E2E) for messages sent between clients.")
        assert picked_def(extr, "E2E") in {"end-to-end"}, extr.picks.get("E2E")

    def test_tier_one_nlp(self):
        det, extr = detect_and_extract(
            "Natural language processing (NLP) is used to detect entities, but the NLP output can be noisy.")
        assert picked_def(extr, "NLP") in {"Natural language processing"}, extr.picks.get("NLP")

    def test_tier_one_ppe_with_parenthetical_tail(self):
        det, extr = detect_and_extract(
            "Personal protective equipment (PPE, required on site) must be worn in the laboratory at all times.")
        assert picked_def(extr, "PPE") in {"Personal protective equipment"}, extr.picks.get("PPE")

    def test_tier_one_ceo(self):
        det, extr = detect_and_extract(
            "The Chief Executive Officer (CEO) approved the new security policy and requested weekly reporting."
        )
        assert picked_def(extr, "CEO") in {"Chief Executive Officer",
                                                 "The Chief Executive Officer"}, extr.picks.get("CEO")

    def test_tier_one_sla_inline_abbreviated_as(self):
        det, extr = detect_and_extract(
            "The Service-level agreement, abbreviated as SLA, defines uptime commitments for the platform."
        )
        assert picked_def(extr, "SLA") in {"Service-level agreement"}, extr.picks.get("SLA")

    def test_tier_one_sla_inline_abbreviated_as_lower_case(self):
        det, extr = detect_and_extract(
            "The service-level agreement, abbreviated as SLA, defines uptime commitments for the platform."
        )
        assert picked_def(extr, "SLA") in {"service-level agreement"}, extr.picks.get("SLA")

    def test_tier_one_jwt_is_extracted_via_sentence_backref(self):
        det, extr = detect_and_extract(
            "We store authentication using JSON Web Tokens. JWT is issued after login and saved in a secure cookie."
        )
        assert picked_def(extr, "JWT") == "JSON Web Tokens", extr.picks.get("JWT")

    def test_tier_one_negative_mismatch_plausible_longform_wrong_acronym(self):
        det, extr = detect_and_extract("Operational readiness (SLA) was reviewed.")
        assert extr.picks.get("SLA") is None, extr.picks.get("SLA")

    def test_tier_one_two_acronyms_same_sentence_both_extracted(self):
        det, extr = detect_and_extract(
            "Natural language processing (NLP) and personal protective equipment (PPE) are mentioned."
        )
        assert picked_def(extr, "NLP") in {"Natural language processing"}, extr.picks.get("NLP")
        assert picked_def(extr, "PPE") in {"personal protective equipment",
                                                 "Personal protective equipment"}, extr.picks.get("PPE")

    def test_tier_one_reverse_parenthetical_longform_before_acronym_if_supported(self):
        det, extr = detect_and_extract("(Portable Document Format) PDF is common.")
        assert picked_def(extr, "PDF") in {"Portable Document Format"}, extr.picks.get("PDF")

    def test_tier_one_ampersand_acronym_key(self):
        det, extr = detect_and_extract("Research & Development (R&D) budgets increased.")
        assert picked_def(extr, "R&D") in {"Research & Development"}, extr.picks.get("R&D")

    def test_tier_one_slash_acronym_key(self):
        det, extr = detect_and_extract("Input/Output (I/O) operations are slow.")
        assert picked_def(extr, "I/O") in {"Input/Output"}, extr.picks.get("I/O")

    def test_tier_one_unicode_apostrophe_in_definition_is_canonicalised(self):
        det, extr = detect_and_extract("Queen’s Award (QA) was announced.")
        assert picked_def(extr, "QA") in {"Queen's Award", "Queen’s Award"}, extr.picks.get("QA")

    def test_tier_one_en_dash_in_definition_preserved(self):
        det, extr = detect_and_extract("Director-General’s Office (DGO) issued guidance.")
        assert picked_def(extr, "DGO") in {"Director-General’s Office",
                                                 "Director-General's Office"}, extr.picks.get("DGO")

    def test_tier_one_all_caps_definition_preserved(self):
        det, extr = detect_and_extract("COST PER ACQUISITION (CPA) is a metric.")
        assert picked_def(extr, "CPA") in {"COST PER ACQUISITION"}, extr.picks.get("CPA")

    def test_tier_one_sentence_backref_does_not_define_json(self):
        det, extr = detect_and_extract(
            "We store authentication using JSON Web Tokens. JSON is widely used. JWT is issued after login."
        )
        assert picked_def(extr, "JWT") in {"JSON Web Tokens"}, extr.picks.get("JWT")
        assert extr.picks.get("JSON") is None, extr.picks.get("JSON")

    def test_tier_one_plural_acronym_surface_normalizes_key(self):
        det, extr = detect_and_extract("We ship PDFs (Portable Document Format) daily.")
        assert "PDF" in det.unique_acronyms
        assert picked_def(extr, "PDF") == "Portable Document Format"

    def test_tier_one_possessive_acronym_surface_normalizes_key(self):
        det, extr = detect_and_extract("A PDF's (Portable Document Format) header is visible.")
        assert "PDF" in det.unique_acronyms
        assert picked_def(extr, "PDF") == "Portable Document Format"


class TestDetectAndExtractE2EConfigAdjustment:

    def test_tier_one_dotted_acronym_key_strips_to_plain_preserves_dots_and_detects(self):
        det, extr = detect_and_extract("The United States of America (U.S.A.) is referenced.",
                                          det_cfg=DetectorConfig(enable_dotted=True, dotted_display="preserve"))
        pprint.pprint(det)
        pprint.pprint(extr)
        assert picked_def(extr, "U.S.A") in {"United States of America"}, extr.picks.get("U.S.A")

    def test_tier_one_dotted_acronym_key_strips_to_plain_removes_dots_and_detects(self):
        det, extr = detect_and_extract("The United States of America (U.S.A.) is referenced.",
                                          det_cfg=DetectorConfig(enable_dotted=True, dotted_display="strip"))
        assert picked_def(extr, "USA") in {"United States of America"}, extr.picks.get("USA")

    def test_tier_one_dotted_acronym_key_strips_to_plain_removes_dots_and_detects_but_not_name_initials(self):
        det, extr = detect_and_extract("The United States of America (U.S.A.) is referenced, written by A.B.",
                                          det_cfg=DetectorConfig(enable_dotted=True, dotted_display="strip"))
        assert picked_def(extr, "USA") in {"United States of America"}, extr.picks.get("USA")
        assert picked_def(extr, "AB") is None
        assert picked_def(extr, "A.B.") is None

    def test_tier_one_dotted_acronym_key_strips_to_plain_preserves_dots_and_detects_but_not_name_initials(self):
        det, extr = detect_and_extract("The United States of America (U.S.A.) is referenced, written by A.B.",
                                          det_cfg=DetectorConfig(enable_dotted=True, dotted_display="preserve"))
        assert picked_def(extr, "U.S.A") in {"United States of America"}, extr.picks.get("U.S.A")
        assert picked_def(extr, "AB") is None
        assert picked_def(extr, "A.B.") is None

    def test_tier_one_dotted_initialism_outside_parentheses_detects_preserve_key(self):
        # Running text, trailing '.' should be stripped by strip_trailing_punct.
        det, extr, r = detect_and_extract(
            "The U.S.A. is referenced.",
            det_cfg=DetectorConfig(enable_dotted=True, dotted_display="preserve"),
            return_reports=True
        )
        pprint.pprint(det)
        # Preserve internal dots, but not terminal punctuation.
        assert picked_def(extr, "U.S.A") is None or True  # definition may not exist in this sentence
        fo = det.unique_acronyms["U.S.A"]
        assert fo.acronym == "U.S.A"
        assert fo.normalized_key == "U.S.A"

        # Occurrence(s) should agree
        assert any(o.normalized_key == "U.S.A" for o in det.occurrences)

    def test_tier_one_dotted_initialism_outside_parentheses_detects_strip_key(self):
        det, extr = detect_and_extract(
            "The U.S.A. is referenced.",
            det_cfg=DetectorConfig(enable_dotted=True, dotted_display="strip"),
        )
        assert "USA" in det.unique_acronyms, det.unique_acronyms
        fo = det.unique_acronyms["USA"]
        assert fo.acronym == "U.S.A"  # surface stays as seen
        assert fo.normalized_key == "USA"  # key is stripped

        assert any(o.normalized_key == "USA" for o in det.occurrences)

    def test_tier_one_dotted_initialism_outside_parentheses_followed_by_comma_detects(self):
        det, extr = detect_and_extract(
            "The U.S.A, as referenced here, is important.",
            det_cfg=DetectorConfig(enable_dotted=True, dotted_display="preserve"),
        )
        assert "U.S.A" in det.unique_acronyms, det.unique_acronyms
        assert det.unique_acronyms["U.S.A"].normalized_key == "U.S.A"

    def test_tier_one_dotted_initialism_outside_parentheses_followed_by_closing_paren_detects(self):
        det, extr = detect_and_extract(
            "This is referenced as U.S.A) in older documents.",
            det_cfg=DetectorConfig(enable_dotted=True, dotted_display="preserve"),
        )
        assert "U.S.A" in det.unique_acronyms, det.unique_acronyms
        assert det.unique_acronyms["U.S.A"].normalized_key == "U.S.A"

    def test_tier_one_two_letter_dotted_whitelist_allows_uk(self):
        det, extr = detect_and_extract(
            "We are based in the U.K. and operate internationally.",
            det_cfg=DetectorConfig(enable_dotted=True, dotted_display="strip"),
        )
        # Whitelisted two-letter dotted should be permitted.
        assert "UK" in det.unique_acronyms, det.unique_acronyms
        assert det.unique_acronyms["UK"].acronym == "U.K"
        assert det.unique_acronyms["UK"].normalized_key == "UK"


    def test_tier_one_two_letter_dotted_not_whitelisted_rejects_name_initials(self):
        det, extr = detect_and_extract(
            "The report was written by A.B. and reviewed by C.D.",
            det_cfg=DetectorConfig(enable_dotted=True, dotted_display="strip"),
        )
        # Non-whitelisted 2-letter dotted tokens should be rejected
        assert "AB" not in det.unique_acronyms
        assert "CD" not in det.unique_acronyms
        assert "A.B." not in det.unique_acronyms
        assert "C.D." not in det.unique_acronyms

    def test_tier_one_preserve_semantics_does_not_keep_terminal_dot_in_key(self):
        det, extr = detect_and_extract(
            "The United States of America (U.S.A.) is referenced.",
            det_cfg=DetectorConfig(enable_dotted=True, dotted_display="preserve"),
        )

        assert "U.S.A" in det.unique_acronyms, det.unique_acronyms
        assert "U.S.A." not in det.unique_acronyms
        assert det.unique_acronyms["U.S.A"].normalized_key == "U.S.A"
        assert picked_def(extr, "U.S.A") in {"United States of America"}, extr.picks.get("U.S.A")
