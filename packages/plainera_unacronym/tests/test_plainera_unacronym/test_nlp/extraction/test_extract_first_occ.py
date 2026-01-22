import re
import pytest
from types import SimpleNamespace as NS


import plainera_unacronym.nlp.extraction.anchored.patterns as mod
import plainera_unacronym.nlp.extraction.anchored.extract as ext

from plainera_unacronym.nlp.extraction import ExtractionConfig
from plainera_unacronym.nlp.extraction.anchored.extract import extract_near_firsts

from plainera_unacronym.nlp.extraction.anchored.patterns import compile_anchored_exact


def _cfg(**overrides):
    base = dict(
        max_phrase_chars=200,
        inline_cues=(r"short\s+for", r"stands?\s+for", r"is\s+(?:an\s+)?acronym\s+for"),
        conf_parenthetical=0.95,
        conf_inline=0.80,
        require_two_words=True,
    )
    base.update(overrides)
    return NS(**base)


def _apply_for_acr(text: str, acr: str, cfg) -> list[tuple[str, float, str, str]]:
    """
    Lightweight harness:
    returns list of (label, conf, acr, def) for all matches across all patterns.
    """
    results = []
    for pat, conf, label in compile_anchored_exact(acr, cfg):
        for m in pat.finditer(text):
            results.append((label, conf, m.group("acr"), m.group("def")))
    return results


class TestCompileAnchoredExact:

    def test_tuple_shape_and_flags(self):
        cfg = _cfg()
        out = compile_anchored_exact("PDF", cfg)

        n_cues = len(cfg.inline_cues)

        expected = (
            3  # fwd + rev + backref
            + n_cues  # inlines_after
            + n_cues  # inlines_before
        )
        assert len(out) == expected

        for pat, conf, label in out:
            assert isinstance(pat, re.Pattern)
            # Flags must include IGNORECASE | MULTILINE
            assert (pat.flags & re.IGNORECASE) == re.IGNORECASE
            assert (pat.flags & re.MULTILINE) == re.MULTILINE
            assert isinstance(conf, float)
            assert label in {"paren_before_acr","def_before", "def_after", "inline", "inline_before"}

    def test_parenthetical_fwd_and_rev_match(self):
        cfg = _cfg()
        text = (
            "Portable Document Format (PDF) is widely used.\n"
            "Also PDF (Portable Document Format) appears later."
        )
        pats = compile_anchored_exact("PDF", cfg)

        fwd = next((p for p in pats if p[2] == "def_before"), None)
        rev = next((p for p in pats if p[2] == "def_after"), None)
        assert fwd and rev

        m1 = fwd[0].search(text)
        m2 = rev[0].search(text)
        assert m1 and m2
        assert m1.group("acr").lower() == "pdf"
        assert m2.group("acr").lower() == "pdf"
        assert "Portable" in m1.group("def")
        assert "Portable" in m2.group("def")

        # confidences are wired correctly
        assert fwd[1] == cfg.conf_parenthetical
        assert rev[1] == cfg.conf_parenthetical

    def test_inline_cues_match(self):
        cfg = _cfg()
        text = (
            "PDF, short for Portable Document Format, is common. "
            "PDF stands for Portable Document Format."
        )
        inlines = [p for p in compile_anchored_exact("PDF", cfg) if p[2] == "inline"]
        assert len(inlines) == len(cfg.inline_cues)

        hits = 0
        for pat, conf, label in inlines:
            for m in pat.finditer(text):
                assert m.group("acr").lower() == "pdf"
                # DEF is non-greedy; only require non-empty capture
                assert m.group("def").strip() != ""
                assert conf == cfg.conf_inline
                hits += 1
        assert hits >= 2  # at least the two examples above

    def test_respects_max_phrase_chars_for_forward(self):
        # Tight limit still matches; DEF capture must be <= max_phrase_chars
        cfg = _cfg(max_phrase_chars=10)
        long_def_text = "Incredibly long descriptive name for a format (PDF)"
        fwd = next((p for p in compile_anchored_exact("PDF", cfg) if p[2] == "def_before"), None)
        assert fwd
        m = fwd[0].search(long_def_text)
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
        pats = compile_anchored_exact("R&D", cfg)
        fwd = next((p for p in pats if p[2] == "def_before"), None)
        assert fwd and fwd[0].search(text)

        # C/A: reverse parenthetical present
        pats = compile_anchored_exact("C/A", cfg)
        rev = next((p for p in pats if p[2] == "def_after"), None)
        m = rev[0].search(text)
        assert m and "Cost per Acquisition" in m.group("def")

        # SME: forward parenthetical present
        pats = compile_anchored_exact("SME", cfg)
        fwd = next((p for p in pats if p[2] == "def_before"), None)
        assert fwd and fwd[0].search(text)

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


def _fo(text: str, acr: str, idx: int):
    """Helper to build a FirstOccurrence-like object for tests."""
    return NS(acronym=acr, start_offset=idx, end_offset=idx + len(acr))



class TestExtractNearFirstsUnit:
    def test_forward_parenthetical_exact_alignment(self, monkeypatch):
        text = "Intro. Portable Document Format (PDF) is common."
        acr = "PDF"
        a0 = text.index("(PDF)") + 1
        fo = _fo(text, acr, a0)

        pat_fwd = re.compile(
            r"\b(?P<def>[^){}]{1,200}?)\s*\(\s*(?P<acr>PDF)\s*\)",
            re.IGNORECASE | re.MULTILINE,
        )

        def fake_compile(_acr, _cfg):
            return ((pat_fwd, 0.95, "def_before"),)

        monkeypatch.setattr(mod, "compile_anchored_exact", fake_compile)

        out = extract_near_firsts(
            text,
            firsts={"PDF": fo},
            window_left=50,
            window_right=50,
            cfg=_cfg(),
        )

        assert out["PDF"] is not None
        assert out["PDF"].definition.strip() != ""
        assert out["PDF"].acr_span == (fo.start_offset, fo.end_offset)
        assert 0 < out["PDF"].confidence <= 0.99
        assert "Portable" in out["PDF"].original_definition

    def test_requires_exact_alignment_other_match_is_ignored(self, monkeypatch):
        text = "PDF appears first. Portable Document Format (PDF) later."
        first_idx = text.index("PDF")
        second_idx = text.rindex("PDF")
        assert first_idx != second_idx

        fo = _fo(text, "PDF", first_idx)

        pat_fwd = re.compile(
            r"\b(?P<def>[^){}]{1,200}?)\s*\(\s*(?P<acr>PDF)\s*\)",
            re.IGNORECASE | re.MULTILINE,
        )

        def fake_compile(_acr, _cfg):
            return ((pat_fwd, 0.95, "def_before"),)

        m2 = pat_fwd.search(text)
        assert m2 is not None
        assert m2.span("acr")[0] == second_idx

        monkeypatch.setattr(mod, "compile_anchored_exact", fake_compile)
        out = extract_near_firsts(text, {"PDF": fo}, window_left=80, window_right=80, cfg=_cfg())

        assert out["PDF"] is None

    def test_inline_match_non_empty_and_confidence_cap(self, monkeypatch):
        text = "PDF stands for Portable Document Format in print."
        a0 = text.index("PDF")
        fo = _fo(text, "PDF", a0)

        pat_inline = re.compile(
            r"\b(?P<acr>PDF)\b\s+stands\s+for\s+(?P<def>[^){}]{1,200}?)",
            re.IGNORECASE | re.MULTILINE,
        )

        def fake_compile(_acr, _cfg):
            return ((pat_inline, 0.995, "inline"),)

        monkeypatch.setattr(ext, "compile_anchored_exact", fake_compile)
        out = extract_near_firsts(text, {"PDF": fo}, window_left=10, window_right=50, cfg=_cfg())

        assert out["PDF"] is not None
        assert out["PDF"].definition.strip() != ""
        assert out["PDF"].acr_span == (a0, a0 + 3)
        assert abs(out["PDF"].confidence - 0.99) < 1e-9

    def test_definition_too_long_is_dropped(self, monkeypatch):
        text = "Extremely verbose explanation that keeps going forever (PDF)"
        a0 = text.index("(PDF)") + 1
        fo = _fo(text, "PDF", a0)

        pat_fwd = re.compile(
            r"\b(?P<def>[^){}]{1,999}?)\s*\(\s*(?P<acr>PDF)\s*\)",
            re.IGNORECASE | re.MULTILINE,
        )

        def fake_compile(_acr, _cfg):
            return ((pat_fwd, 0.95, "def_before"),)

        monkeypatch.setattr(mod, "compile_anchored_exact", fake_compile)

        cfg = _cfg(max_phrase_chars=8)
        out = extract_near_firsts(text, {"PDF": fo}, window_left=50, window_right=50, cfg=cfg)

        assert out["PDF"] is None

    def test_reverse_parenthetical_ca_full_phrase(self, monkeypatch):
        text = "The CFO said C/A (Cost per Acquisition) has fallen."
        acr = "C/A"
        a0 = text.index("C/A")
        fo = _fo(text, acr, a0)

        pat_rev = re.compile(
            r"\b(?P<acr>C\/A)\s*\(\s*(?P<def>[^){}]{1,200}?)\s*\)",
            re.IGNORECASE | re.MULTILINE,
        )

        def fake_compile(_acr, _cfg):
            return ((pat_rev, 0.95, "def_after"),)

        monkeypatch.setattr(mod, "compile_anchored_exact", fake_compile)
        out = extract_near_firsts(text, {"C/A": fo}, window_left=40, window_right=60, cfg=_cfg())

        assert out["C/A"] is not None
        assert out["C/A"].definition == "Cost per Acquisition"
        assert out["C/A"].original_definition == "Cost per Acquisition"
        assert out["C/A"].acr_span == (a0, a0 + 3)
        assert 0 < out["C/A"].confidence <= 0.99

def _cfg_near_firsts_integrated(**overrides):
    # Realistic default config (only fields used here matter)
    return ExtractionConfig(**{
        "inline_cues": (
            r"short\s+for",
            r"stands?\s+for",
            r"is\s+(?:an\s+)?acronym\s+for",
        ),
        "max_phrase_chars": overrides.get("max_phrase_chars", 200),
        "enabled_parenthetical": True,
        "enabled_inline": True,
        "conf_parenthetical": 0.95,
        "conf_inline": 0.80,
    })


class TestExtractNearFirstsIntegration:
    def test_mixed_forward_reverse_inline(self):
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
            "R&D": _fo(text, "R&D", r_and_d_idx),
            "C/A": _fo(text, "C/A", c_a_idx),
            "PTO": _fo(text, "PTO", pto_idx),
            "AM": _fo(text, "AM", am_idx),
            "PDF": _fo(text, "PDF", pdf_idx),
        }

        out = extract_near_firsts(text, firsts, window_left=80, window_right=120, cfg=_cfg_near_firsts_integrated())

        # R&D forward parenthetical
        assert out["R&D"] is not None
        assert "Research" in out["R&D"].definition
        assert out["R&D"].acr_span == (r_and_d_idx, r_and_d_idx + 3)
        assert 0 < out["R&D"].confidence <= 0.99

        # C/A reverse parenthetical
        assert out["C/A"] is not None
        assert out["C/A"].definition == "Cost per Acquisition"
        assert out["C/A"].original_definition == "Cost per Acquisition"
        assert out["C/A"].acr_span == (c_a_idx, c_a_idx + 3)
        assert 0 < out["C/A"].confidence <= 0.99

        # PTO inline (“stands for …”)
        assert out["PTO"] is not None
        assert out["PTO"].definition.strip() != ""
        assert out["PTO"].acr_span == (pto_idx, pto_idx + 3)
        assert 0 < out["PTO"].confidence <= 0.99

        # AM inline (“short for …”)
        assert out["AM"] is not None
        assert out["AM"].definition.strip() != ""
        assert out["AM"].acr_span == (am_idx, am_idx + 2)
        assert 0 < out["AM"].confidence <= 0.99

        # PDF (first is forward)
        assert out["PDF"] is not None
        assert "Portable" in out["PDF"].definition
        assert out["PDF"].acr_span == (pdf_idx, pdf_idx + 3)

    def test_length_gating_and_normalisation_on_long_inline(self):
        # Very long inline tail; max_phrase_chars should gate the capture
        text = (
            "In printing, PTO stands for "
            "a very, very long descriptive phrase that should be trimmed or rejected entirely "
            "depending on configuration and normalisation steps."
        )
        pto_idx = text.index("PTO")
        firsts = {"PTO": _fo(text, "PTO", pto_idx)}

        cfg_strict = ExtractionConfig(
            inline_cues=(r"short\s+for", r"stands?\s+for", r"is\s+(?:an\s+)?acronym\s+for"),
            max_phrase_chars=20,
            enabled_parenthetical=True,
            enabled_inline=True,
            conf_parenthetical=0.95,
            conf_inline=0.80,
            require_two_words=False,
        )
        out_strict = extract_near_firsts(text, firsts, window_left=10, window_right=200, cfg=cfg_strict)
        print(out_strict)
        assert out_strict["PTO"] is None

        cfg_relaxed = ExtractionConfig(
            inline_cues=(r"short\s+for", r"stands?\s+for", r"is\s+(?:an\s+)?acronym\s+for"),
            max_phrase_chars=160,
            enabled_parenthetical=True,
            enabled_inline=True,
            conf_parenthetical=0.95,
            conf_inline=0.80,
        )
        out_relaxed = extract_near_firsts(text, firsts, window_left=10, window_right=200, cfg=cfg_relaxed)
        assert out_relaxed["PTO"] is not None
        assert out_relaxed["PTO"].definition.strip() != ""
        assert out_relaxed["PTO"].acr_span == (pto_idx, pto_idx + 3)
        assert 0 < out_relaxed["PTO"].confidence <= 0.99

    def test_ignores_matches_not_aligned_to_first_occurrence(self):
        text = "PDF appears first. Portable Document Format (PDF) later still."
        first_pdf_idx = text.index("PDF")
        firsts = {"PDF": _fo(text, "PDF", first_pdf_idx)}

        out = extract_near_firsts(text, firsts, window_left=80, window_right=80, cfg=_cfg())

        assert out["PDF"] is None
