import re
import pytest
from types import SimpleNamespace as NS

from plainera_unacronym.nlp.extraction.extract_first_occ import _compile_anchored_exact


def _cfg(**overrides):
    base = dict(
        max_phrase_chars=200,
        inline_cues=(r"short\s+for", r"stands?\s+for", r"is\s+(?:an\s+)?acronym\s+for"),
        conf_parenthetical=0.95,
        conf_inline=0.80,
    )
    base.update(overrides)
    return NS(**base)


def _apply_for_acr(text: str, acr: str, cfg) -> list[tuple[str, float, str, str]]:
    """
    Lightweight harness:
    returns list of (label, conf, acr, def) for all matches across all patterns.
    """
    results = []
    for pat, conf, label in _compile_anchored_exact(acr, cfg):
        for m in pat.finditer(text):
            results.append((label, conf, m.group("acr"), m.group("def")))
    return results


class TestCompileAnchoredExact:
    def test_tuple_shape_and_flags(self):
        cfg = _cfg()
        out = _compile_anchored_exact("PDF", cfg)

        # 2 parenthetical + len(inline_cues)
        assert len(out) == 2 + len(cfg.inline_cues)

        for pat, conf, label in out:
            assert isinstance(pat, re.Pattern)
            # Flags must include IGNORECASE | MULTILINE
            assert (pat.flags & re.IGNORECASE) == re.IGNORECASE
            assert (pat.flags & re.MULTILINE) == re.MULTILINE
            assert isinstance(conf, float)
            assert label in {"def_before", "def_after", "inline"}

    def test_parenthetical_fwd_and_rev_match(self):
        cfg = _cfg()
        text = (
            "Portable Document Format (PDF) is widely used.\n"
            "Also PDF (Portable Document Format) appears later."
        )
        pats = _compile_anchored_exact("PDF", cfg)

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
        inlines = [p for p in _compile_anchored_exact("PDF", cfg) if p[2] == "inline"]
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
        fwd = next((p for p in _compile_anchored_exact("PDF", cfg) if p[2] == "def_before"), None)
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
        pats = _compile_anchored_exact("R&D", cfg)
        fwd = next((p for p in pats if p[2] == "def_before"), None)
        assert fwd and fwd[0].search(text)

        # C/A: reverse parenthetical present
        pats = _compile_anchored_exact("C/A", cfg)
        rev = next((p for p in pats if p[2] == "def_after"), None)
        m = rev[0].search(text)
        assert m and "Cost per Acquisition" in m.group("def")

        # SME: forward parenthetical present
        pats = _compile_anchored_exact("SME", cfg)
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
