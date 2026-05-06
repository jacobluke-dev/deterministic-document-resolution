import re
from types import SimpleNamespace as NS

import document_resolution.nlp.extraction.acronyms.anchored.extract as ext
import document_resolution.nlp.extraction.acronyms.anchored.patterns as mod
from document_resolution.nlp.extraction.acronyms.anchored.extract import (
    compile_anchored_for_surface,
    extract_near_firsts,
)
from document_resolution.nlp.extraction.acronyms.config import ExtractionConfig


def _cfg(**overrides):
    base = dict(
        max_phrase_chars=200,
        inline_cues=(r"short\s+for", r"stands?\s+for", r"is\s+(?:an\s+)?acronym\s+for"),
        conf_parenthetical=0.95,
        conf_inline=0.80,
        require_two_words=True,
    )
    base.update(overrides)
    conf = NS(
        base_by_source={
            "parenthetical": base["conf_parenthetical"],
            "inline": base["conf_inline"],
            "first_occurrence_anchored": 0.85,
        }
    )
    base["confidence"] = conf
    return NS(**base)


def _apply_for_acr(text: str, acr: str, cfg) -> list[tuple[str, float, str, str]]:
    """
    Lightweight harness:
    returns list of (label, conf, acr, def) for all matches across all patterns.
    """
    results = []
    for spec in compile_anchored_for_surface(acr, cfg):
        for m in spec.pat.finditer(text):
            results.append((spec.kind, spec.base_conf, m.group("acr"), m.group("def")))
    return results


class TestCompileAnchoredExact:
    def test_tuple_shape_and_flags(self):
        cfg = _cfg()
        out = compile_anchored_for_surface("PDF", cfg)

        n_cues = len(cfg.inline_cues)

        expected = (
            6  # b4 + dir etc + fwd + rev + backref
            + n_cues  # inlines_after
            + n_cues  # inlines_before
        )
        assert len(out) == expected
        spec = out
        pat = spec[0].pat
        base_conf = spec[0].base_conf
        kind = spec[0].kind
        strategy = spec[0].strategy

        assert isinstance(pat, re.Pattern)
        # Flags must include IGNORECASE | MULTILINE
        assert (pat.flags & re.IGNORECASE) == re.IGNORECASE
        assert (pat.flags & re.MULTILINE) == re.MULTILINE
        assert isinstance(base_conf, float)
        assert kind in {
            "before_acr_paren",
            "def_after_direct",
            "def_before_direct",
            "paren_before_acr",
            "def_before",
            "def_after",
            "inline",
            "inline_before",
        }
        assert strategy in {"direct_def", "helper_def_before", "helper_inline_after", "helper_def_after"}

    def test_parenthetical_fwd_and_rev_match(self):
        cfg = _cfg()
        text = "Portable Document Format (PDF) is widely used.\n" "Also PDF (Portable Document Format) appears later."
        pats = compile_anchored_for_surface("PDF", cfg)

        fwd = next((p for p in pats if p.kind == "def_before"), None)
        rev = next((p for p in pats if p.kind == "def_after"), None)
        assert fwd and rev

        m1 = fwd.pat.search(text)
        m2 = rev.pat.search(text)
        assert m1 and m2
        assert m1.group("acr").lower() == "pdf"
        assert m2.group("acr").lower() == "pdf"
        assert "Portable" in m1.group("def")
        assert "Portable" in m2.group("def")

        # confidences are wired correctly
        assert fwd.base_conf == cfg.conf_parenthetical
        assert rev.base_conf == cfg.conf_parenthetical

    def test_inline_cues_match(self):
        cfg = _cfg()
        text = "PDF, short for Portable Document Format, is common. " "PDF stands for Portable Document Format."
        inlines = [p for p in compile_anchored_for_surface("PDF", cfg) if p.kind == "inline"]
        assert len(inlines) == len(cfg.inline_cues)

        hits = 0
        for spec in inlines:
            for m in spec.pat.finditer(text):
                assert m.group("acr").lower() == "pdf"
                # DEF is non-greedy; only require non-empty capture
                assert m.group("def").strip() != ""
                assert spec.base_conf == cfg.conf_inline
                hits += 1
        assert hits >= 2  # at least the two examples above

    def test_respects_max_phrase_chars_for_forward(self):
        # Tight limit still matches; DEF capture must be <= max_phrase_chars
        cfg = _cfg(max_phrase_chars=10)
        long_def_text = "Incredibly long descriptive name for a format (PDF)"
        fwd = next((p for p in compile_anchored_for_surface("PDF", cfg) if p.kind == "def_before"), None)
        assert fwd
        m = fwd.pat.search(long_def_text)
        assert m is not None
        assert len(m.group("def")) <= 10

    def test_escapes_special_chars_in_acronym(self):
        # Ensure re.escape(acr) works: &, / and - are common in acronyms
        cfg = _cfg()
        text = (
            "Research and Development (R&D) fuels innovation. "
            "We track cost of acquisition: C/A (Cost per Acquisition). "
            "Small-to-medium Enterprises (SME) are numerous."
        )

        # R&D: forward parenthetical present
        pats = compile_anchored_for_surface("R&D", cfg)
        fwd = next((p for p in pats if p.kind == "def_before"), None)
        assert fwd and fwd.pat.search(text)

        # C/A: reverse parenthetical present
        pats = compile_anchored_for_surface("C/A", cfg)
        rev = next((p for p in pats if p.kind == "def_after"), None)
        m = rev.pat.search(text)
        assert m and "Cost per Acquisition" in m.group("def")

        # SME: forward parenthetical present
        pats = compile_anchored_for_surface("SME", cfg)
        fwd = next((p for p in pats if p.kind == "def_before"), None)
        assert fwd and fwd.pat.search(text)

    def test_mixed_forms_multiple_acronyms_across_text(self):
        cfg = _cfg()
        text = (
            "We invest in Research and Development (R&D) to innovate.\n"
            "The CFO said C/A (Cost per Acquisition) has fallen this quarter.\n"
            "PTO stands for Please Turn Over on print jobs.\n"
            "Finally, AM, short for amplitude modulation, is a legacy technique."
        )

        # R&D: forward parenthetical
        r_and_d = _apply_for_acr(text, "R&D", cfg)
        assert any(lbl == "def_before" and "Research and Development" in d for lbl, _, _, d in r_and_d)

        # C/A: reverse parenthetical
        ca = _apply_for_acr(text, "C/A", cfg)
        assert any(lbl == "def_after" and "Cost per Acquisition" in d for lbl, _, _, d in ca)

        # PTO: inline cue exists with non-empty DEF
        pto = _apply_for_acr(text, "PTO", cfg)
        assert any(lbl == "inline" and d.strip() != "" for lbl, _, _, d in pto)

        # AM: inline "short for" with non-empty DEF
        am = _apply_for_acr(text, "AM", cfg)
        assert any(lbl == "inline" and d.strip() != "" for lbl, _, _, d in am)

        # Confidence wiring sanity: parenthetical vs inline
        assert any(conf == cfg.conf_parenthetical for lbl, conf, _, _ in r_and_d + ca)
        assert any(conf == cfg.conf_inline for lbl, conf, _, _ in pto + am)

    def test_multiline_and_case_insensitive_matching(self):
        cfg = _cfg()
        text = (
            "portable document format (pdf)\n"
            "PDF (Portable Document Format)\n"
            "Pdf, is an acronym for portable document format\n"
        )

        hits = _apply_for_acr(text, "PDF", cfg)
        assert len(hits) >= 3
        # All should capture the same acronym text (case-insensitive group)
        assert all(h[2].lower() == "pdf" for h in hits)
        # Each hit must have a non-empty definition
        assert all(h[3].strip() for h in hits)



class TestExtractNearFirstsUnit:
    def test_forward_parenthetical_exact_alignment(self, _patch, fo):
        text = "Intro. Portable Document Format (PDF) is common."
        acr = "PDF"
        a0 = text.index("(PDF)") + 1
        first_occurrence = fo(acr, a0, a0 + len(acr))

        pat_fwd = re.compile(
            r"\b(?P<def>[^){}]{1,200}?)\s*\(\s*(?P<acr>PDF)\s*\)",
            re.IGNORECASE | re.MULTILINE,
        )

        def fake_compile(_acr, _cfg):
            pattern_spec = mod.PatternSpec(pat=pat_fwd,
                                           base_conf=0.95,
                                           strategy="helper_def_before",
                                           kind="def_before"),
            return pattern_spec

        _patch(extract_near_firsts, compile_anchored_for_surface=fake_compile)

        _patch(
            extract_near_firsts,
            compile_anchored_for_surface=fake_compile,
            base_for_kind=lambda _cfg, kind: 0.995 if kind in {"inline", "inline_before"} else 0.95,
        )


        out = extract_near_firsts(
            text,
            firsts={"PDF": first_occurrence},
            window_left=50,
            window_right=50,
            cfg=_cfg(),
        )

        assert out["PDF"] is not None
        assert out["PDF"].definition.strip() != ""
        assert out["PDF"].acr_span == (first_occurrence.start_offset, first_occurrence.end_offset)
        assert 0 < out["PDF"].definition_confidence <= 0.99
        assert "Portable" in out["PDF"].original_definition

    def test_requires_exact_alignment_other_match_is_ignored(self, _patch, fo):
        text = "PDF appears first. Portable Document Format (PDF) later."
        a0 = text.index("PDF")
        a1 = text.rindex("PDF")
        acr = "PDF"
        assert a0 != a1

        first_occurrence = fo(acr, a0, a0 + len(acr))

        pat_fwd = re.compile(
            r"\b(?P<def>[^){}]{1,200}?)\s*\(\s*(?P<acr>PDF)\s*\)",
            re.IGNORECASE | re.MULTILINE,
        )

        def fake_compile(_acr, _cfg):
            pattern_spec = mod.PatternSpec(pat=pat_fwd,
                                           base_conf=0.95,
                                           strategy="helper_def_before",
                                           kind="def_before"),
            return pattern_spec

        m2 = pat_fwd.search(text)
        assert m2 is not None
        assert m2.span("acr")[0] == a1

        _patch(extract_near_firsts, compile_anchored_for_surface=fake_compile)
        out = extract_near_firsts(text, {"PDF": first_occurrence}, window_left=80, window_right=80, cfg=_cfg())

        assert out["PDF"] is None

    def test_inline_match_non_empty_and_confidence_cap(self, _patch, fo):
        text = "PDF stands for Portable Document Format in print."
        acr = "PDF"
        a0 = text.index("PDF")
        first_occurrence = fo(acr, a0, a0 + len(acr))

        pat_inline = re.compile(
            r"\b(?P<acr>PDF)\b\s+stands\s+for\s+(?P<def>[^){}]{1,200}?)",
            re.IGNORECASE | re.MULTILINE,
        )

        def fake_compile(_acr, _cfg):
            return (
                mod.PatternSpec(
                    pat=pat_inline,
                    base_conf=0.995,
                    strategy="helper_inline_after",
                    kind="inline",
                ),
            )

        _patch(
            extract_near_firsts,
            compile_anchored_for_surface=fake_compile,
            base_for_kind=lambda _cfg, kind: 0.995 if kind in {"inline", "inline_before"} else 0.95,
        )

        out = extract_near_firsts(text, {"PDF": first_occurrence}, window_left=10, window_right=50, cfg=_cfg())

        assert out["PDF"] is not None
        assert out["PDF"].definition.strip() != ""
        assert out["PDF"].acr_span == (a0, a0 + 3)
        assert abs(out["PDF"].definition_confidence - 0.99) < 1e-9

    def test_definition_too_long_is_dropped(self, _patch, fo):
        text = "Extremely verbose explanation that keeps going forever (PDF)"
        acr = "PDF"
        a0 = text.index("(PDF)") + 1
        first_occurrence = fo(acr, a0, a0 + len(acr))

        pat_fwd = re.compile(
            r"\b(?P<def>[^){}]{1,999}?)\s*\(\s*(?P<acr>PDF)\s*\)",
            re.IGNORECASE | re.MULTILINE,
        )

        def fake_compile(_acr, _cfg):
            pattern_spec = mod.PatternSpec(pat=pat_fwd, base_conf=0.95, strategy="helper_def_before", kind="def_before"),
            return pattern_spec

        _patch(extract_near_firsts, compile_anchored_for_surface=fake_compile)

        cfg = _cfg(max_phrase_chars=8)
        out = extract_near_firsts(text, {"PDF": first_occurrence}, window_left=50, window_right=50, cfg=cfg)

        assert out["PDF"] is None

    def test_reverse_parenthetical_ca_full_phrase(self, _patch, fo):
        text = "The CFO said C/A (Cost per Acquisition) has fallen."
        acr = "C/A"
        a0 = text.index("C/A")
        first_occurrence = fo(acr, a0, a0 + len(acr))

        pat_rev = re.compile(
            r"\b(?P<acr>C\/A)\s*\(\s*(?P<def>[^){}]{1,200}?)\s*\)",
            re.IGNORECASE | re.MULTILINE,
        )

        def fake_compile(_acr, _cfg):
            pattern_spec = mod.PatternSpec(pat=pat_rev,
                                           base_conf=0.95,
                                           strategy="helper_def_after",
                                           kind="def_after"),
            return pattern_spec

        _patch(
            extract_near_firsts,
            compile_anchored_for_surface=fake_compile,
            base_for_kind=lambda _cfg, kind: 0.95 if kind in {"inline", "def_after"} else 0.95,
        )

        out = extract_near_firsts(text, {"C/A": first_occurrence}, window_left=40, window_right=60, cfg=_cfg())

        assert out["C/A"] is not None
        assert out["C/A"].definition == "Cost per Acquisition"
        assert out["C/A"].original_definition == "Cost per Acquisition"
        assert out["C/A"].acr_span == (a0, a0 + 3)
        assert 0 < out["C/A"].definition_confidence <= 0.99


def _cfg_near_firsts_integrated(**overrides):
    # Realistic default config (only fields used here matter)
    return ExtractionConfig(
        **{
            "inline_cues": (
                r"short\s+for",
                r"stands?\s+for",
                r"is\s+(?:an\s+)?acronym\s+for",
            ),
            "max_phrase_chars": overrides.get("max_phrase_chars", 200),
        }
    )


class TestExtractNearFirstsIntegration:
    def test_mixed_forward_reverse_inline(self, fo):
        text = (
            "We invest in Research and Development (R&D) to innovate.\n"
            "The CFO said C/A (Cost per Acquisition) has fallen this quarter.\n"
            "PTO stands for Please Turn Over on print jobs.\n"
            "Finally, AM, short for amplitude modulation, is a legacy technique.\n"
            "Portable Document Format (PDF) dominates documents; elsewhere PDF (Portable Document Format) appears."
        )

        r_and_d_idx = text.index("(R&D)") + 1
        c_a_idx = text.index("C/A")
        pto_idx = text.index("PTO")
        am_idx = text.index("AM, short")
        pdf_idx = text.index("(PDF)") + 1

        firsts = {
            "R&D": fo("R&D", r_and_d_idx, r_and_d_idx + 3),
            "C/A": fo("C/A", c_a_idx, c_a_idx + 3),
            "PTO": fo("PTO", pto_idx, pto_idx + 3),
            "AM": fo("AM", am_idx, am_idx + 2),
            "PDF": fo("PDF", pdf_idx, pdf_idx + 3),
        }

        out = extract_near_firsts(
            text,
            firsts,
            window_left=80,
            window_right=120,
            cfg=_cfg_near_firsts_integrated(),
        )

        # R&D forward parenthetical
        assert out["R&D"] is not None
        assert "Research" in out["R&D"].definition
        assert out["R&D"].acr_span == (r_and_d_idx, r_and_d_idx + 3)
        assert 0 < out["R&D"].definition_confidence <= 0.99

        # C/A reverse parenthetical
        assert out["C/A"] is not None
        assert out["C/A"].definition == "Cost per Acquisition"
        assert out["C/A"].original_definition == "Cost per Acquisition"
        assert out["C/A"].acr_span == (c_a_idx, c_a_idx + 3)
        assert 0 < out["C/A"].definition_confidence <= 0.99

        # PTO inline (“stands for …”)
        assert out["PTO"] is not None
        assert out["PTO"].definition.strip() != ""
        assert out["PTO"].acr_span == (pto_idx, pto_idx + 3)
        assert 0 < out["PTO"].definition_confidence <= 0.99

        # AM inline (“short for …”)
        assert out["AM"] is not None
        assert out["AM"].definition.strip() != ""
        assert out["AM"].acr_span == (am_idx, am_idx + 2)
        assert 0 < out["AM"].definition_confidence <= 0.99

        # PDF (first is forward)
        assert out["PDF"] is not None
        assert "Portable" in out["PDF"].definition
        assert out["PDF"].acr_span == (pdf_idx, pdf_idx + 3)

    def test_length_gating_and_normalisation_on_long_inline(self, fo):
        # Very long inline tail; max_phrase_chars should gate the capture
        text = (
            "In printing, PTO stands for "
            "a very, very long descriptive phrase that should be trimmed or rejected entirely "
            "depending on configuration and normalisation steps."
        )
        pto_idx = text.index("PTO")
        acr = "PTO"
        first_occurrence = {"PTO": fo(acr, pto_idx, pto_idx + len(acr))}

        cfg_strict = ExtractionConfig(
            inline_cues=(r"short\s+for", r"stands?\s+for", r"is\s+(?:an\s+)?acronym\s+for"),
            max_phrase_chars=20,
            require_two_words=False,
        )
        out_strict = extract_near_firsts(text, first_occurrence, window_left=10, window_right=200, cfg=cfg_strict)
        print(out_strict)
        assert out_strict["PTO"] is None

        cfg_relaxed = ExtractionConfig(
            inline_cues=(r"short\s+for", r"stands?\s+for", r"is\s+(?:an\s+)?acronym\s+for"),
            max_phrase_chars=160,
        )
        out_relaxed = extract_near_firsts(text, first_occurrence, window_left=10, window_right=200, cfg=cfg_relaxed)
        assert out_relaxed["PTO"] is not None
        assert out_relaxed["PTO"].definition.strip() != ""
        assert out_relaxed["PTO"].acr_span == (pto_idx, pto_idx + 3)
        assert 0 < out_relaxed["PTO"].definition_confidence <= 0.99

    def test_ignores_matches_not_aligned_to_first_occurrence(self, fo):
        text = "PDF appears first. Portable Document Format (PDF) later still."
        acr = "PDF"
        firsts = {"PDF": fo("PDF", 0, len(acr))}

        out = extract_near_firsts(text, firsts, window_left=80, window_right=80, cfg=_cfg())

        assert out["PDF"] is None
