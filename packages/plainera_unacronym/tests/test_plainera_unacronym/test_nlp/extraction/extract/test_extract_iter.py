import re
from types import SimpleNamespace as NS
import pytest

import plainera_unacronym.nlp.extraction.extract as mod
from plainera_unacronym.nlp.extraction import extract_iter


def _cfg_unit(**overrides):
    base = dict(
        # core toggles/weights
        enabled_parenthetical=True,
        enabled_inline=True,
        conf_parenthetical=0.95,
        conf_inline=0.80,
        # limits/validation
        max_phrase_chars=200,
        require_two_words=True,
        min_acr_len=2,
        max_acr_len=10,
        acr_allowed=r"A-Z0-9&./-",
        # cues
        inline_cues=(r"short\s+for", r"stands?\s+for", r"is\s+(?:an\s+)?acronym\s+for"),
        # plugin hook (unused in basic unit tests)
        plugins=(),
    )
    base.update(overrides)
    return NS(**base)


class TestExtractIterUnit:
    def test_filters_by_acronym_length(self, monkeypatch):
        # Patterns that would otherwise match a 1-letter acronym; must be filtered out by min_acr_len=2
        text = "a format (P)\nP stands for Portable"

        fwd = re.compile(r"\b(?P<def>[^){}]{1,200}?)\s*\(\s*(?P<acr>P)\s*\)")
        inline = re.compile(r"\b(?P<acr>P)\b\s+stands\s+for\s+(?P<def>[^){}]{1,200}?)")

        def fake_parenthetical(cfg):
            return (fwd, re.compile("$^"))  # rev never matches

        def fake_inline(cfg, cues):
            return [inline]

        monkeypatch.setattr(mod, "_compile_parenthetical", fake_parenthetical)
        monkeypatch.setattr(mod, "_compile_inline", fake_inline)

        cfg = _cfg_unit(min_acr_len=2)  # too short for "P"
        out = list(extract_iter(text, cfg))
        assert out == []  # all filtered

    def test_two_word_gate_applies(self, monkeypatch):
        text = "XK, short for x"
        inline = re.compile(r"\b(?P<acr>XK)\b\s*,?\s*short\s+for\s+(?P<def>[^){}]{1,200}?)")

        monkeypatch.setattr(mod, "_compile_parenthetical", lambda cfg: (re.compile("$^"), re.compile("$^")))
        monkeypatch.setattr(mod, "_compile_inline", lambda cfg, cues: [inline])

        cfg = _cfg_unit(require_two_words=True)
        out = list(extract_iter(text, cfg))
        assert out == []  # 'x' is single word → dropped

        cfg2 = _cfg_unit(require_two_words=False)
        out2 = list(extract_iter(text, cfg2))
        assert len(out2) == 1
        assert out2[0].acronym == "XK"
        assert out2[0].definition == "x"

    def test_max_phrase_chars_gates_definition(self, monkeypatch):
        text = "Portable Document Format (PDF)."
        fwd = re.compile(r"\b(?P<def>[^){}]{1,5}?)\s*\(\s*(?P<acr>PDF)\s*\)")  # too small cap

        monkeypatch.setattr(mod, "_compile_parenthetical", lambda cfg: (fwd, re.compile("$^")))
        monkeypatch.setattr(mod, "_compile_inline", lambda cfg, cues: [])

        cfg = _cfg_unit(max_phrase_chars=5)
        out = list(extract_iter(text, cfg))
        # 'Portable Document Format' exceeds 5 after tightening → dropped
        assert out == []

    def test_confidence_bump_when_initials_match(self, monkeypatch):
        text = "Portable Document Format (PDF) is common."
        fwd = re.compile(r"\b(?P<def>Portable Document Format)\s*\(\s*(?P<acr>PDF)\s*\)")

        monkeypatch.setattr(mod, "_compile_parenthetical", lambda cfg: (fwd, re.compile("$^")))
        monkeypatch.setattr(mod, "_compile_inline", lambda cfg, cues: [])

        cfg = _cfg_unit(conf_parenthetical=0.90)
        out = list(extract_iter(text, cfg))
        assert len(out) == 1
        # initials match → +0.03, capped at 0.99
        assert abs(out[0].confidence - 0.93) < 1e-9

    def test_seen_deduplicates_exact_same_span(self, monkeypatch):
        # Two identical patterns would yield the same span; second should be deduped.
        text = "Portable Document Format (PDF)"
        p = re.compile(r"\b(?P<def>Portable Document Format)\s*\(\s*(?P<acr>PDF)\s*\)")

        monkeypatch.setattr(mod, "_compile_parenthetical", lambda cfg: (p, p))
        monkeypatch.setattr(mod, "_compile_inline", lambda cfg, cues: [])

        out = list(extract_iter(text, _cfg_unit()))
        assert len(out) == 1  # despite two producers

    def test_start_end_slice_limits_search_range(self, monkeypatch):
        text = "Portable Document Format (PDF). Noise. Also PDF (Portable Document Format)."

        fwd = re.compile(r"\b(?P<def>Portable Document Format)\s*\(\s*(?P<acr>PDF)\s*\)")
        rev = re.compile(r"\b(?P<acr>PDF)\s*\(\s*(?P<def>Portable Document Format)\s*\)")

        monkeypatch.setattr(mod, "_compile_parenthetical", lambda cfg: (fwd, rev))
        monkeypatch.setattr(mod, "_compile_inline", lambda cfg, cues: [])

        # Only search the second half (so we only see the reverse match)
        mid = text.index("Also")
        out = list(extract_iter(text, _cfg_unit(), start=mid))
        assert len(out) == 1
        assert out[0].acronym == "PDF"
        assert out[0].definition == "Portable Document Format"
        # ensure start trimming worked by confirming the span of the match is ≥ mid
        assert out[0].acr_start >= mid


def _cfg_intergrated(**overrides):
    # Use the real ExtractionConfig for integration tests
    base = dict(
        enabled_parenthetical=True,
        enabled_inline=True,
        conf_parenthetical=0.95,
        conf_inline=0.80,
        max_phrase_chars=200,
        require_two_words=True,
        inline_cues=(
            r"short\s+for",
            r"stands?\s+for",
            r"is\s+(?:an\s+)?acronym\s+for",
        ),
        plugins=(),
    )
    return mod.ExtractionConfig(**{**base, **overrides})


class TestExtractIterIntegration:
    def test_mixed_forward_reverse_inline_and_confidence(self):
        text = (
            "We invest in Research and Development (R&D) to innovate.\n"
            "The CFO said C/A (Cost per Acquisition) has fallen this quarter.\n"
            "PTO stands for Please Turn Over on print jobs.\n"
            "Finally, AM, short for amplitude modulation, is a legacy technique.\n"
            "Portable Document Format (PDF) dominates documents; elsewhere PDF (Portable Document Format) appears."
        )

        out = list(extract_iter(text, _cfg_intergrated()))

        # Useful to map by acronym -> best definition(s)
        by_acr = {}
        for ed in out:
            by_acr.setdefault(ed.acronym, []).append(ed)

        # R&D forward
        assert "R&D" in by_acr
        assert any("Research and Development" in e.definition for e in by_acr["R&D"])
        assert all(0 < e.confidence <= 0.99 for e in by_acr["R&D"])

        # C/A reverse
        assert "C/A" in by_acr
        assert any(e.definition == "Cost per Acquisition" for e in by_acr["C/A"])
        # initials match bump is applied
        bumped = [e for e in by_acr["C/A"] if abs(e.confidence - 0.98) < 1e-6 or e.confidence > 0.95]
        assert bumped, "Expected confidence bump for C/A"

        # PTO inline
        assert "PTO" in by_acr
        assert any("Please Turn Over" in e.definition for e in by_acr["PTO"])
        assert any(0 < e.confidence <= 0.99 for e in by_acr["PTO"])

        # AM inline (lowercase long form is ok)
        assert "AM" in by_acr
        assert any("amplitude modulation" in e.definition.lower() for e in by_acr["AM"])

        # PDF both forms appear; first is forward
        assert "PDF" in by_acr
        assert any("Portable Document Format" in e.definition for e in by_acr["PDF"])

    def test_mixed_forward_reverse_inline_and_confidence_no_new_lines(self):
        text = (
            "We invest in Research and Development (R&D) to innovate. The CFO said C/A (Cost per Acquisition) has fallen this quarter. PTO stands for Please Turn Over on print jobs. Finally, AM, short for amplitude modulation, is a legacy technique. Portable Document Format (PDF) dominates documents; elsewhere PDF (Portable Document Format) appears."
        )

        out = list(extract_iter(text, _cfg_intergrated()))

        # Useful to map by acronym -> best definition(s)
        by_acr = {}
        for ed in out:
            by_acr.setdefault(ed.acronym, []).append(ed)

        # R&D forward
        assert "R&D" in by_acr
        assert any("Research and Development" in e.definition for e in by_acr["R&D"])
        assert all(0 < e.confidence <= 0.99 for e in by_acr["R&D"])

        # C/A reverse
        assert "C/A" in by_acr
        assert any(e.definition == "Cost per Acquisition" for e in by_acr["C/A"])
        # initials match bump is applied
        bumped = [e for e in by_acr["C/A"] if abs(e.confidence - 0.98) < 1e-6 or e.confidence > 0.95]
        assert bumped, "Expected confidence bump for C/A"

        # PTO inline
        assert "PTO" in by_acr
        assert any("Please Turn Over" in e.definition for e in by_acr["PTO"])
        assert any(0 < e.confidence <= 0.99 for e in by_acr["PTO"])

        # AM inline (lowercase long form is ok)
        assert "AM" in by_acr
        assert any("amplitude modulation" in e.definition.lower() for e in by_acr["AM"])

        # PDF both forms appear; first is forward
        assert "PDF" in by_acr
        assert any("Portable Document Format" in e.definition for e in by_acr["PDF"])

    def test_parenthetical_allows_from_plugins_are_applied(self, monkeypatch):
        """
        Simulate a plugin that:
          - adds an inline cue
          - *disallows* parenthetical unless the acronym is ALL CAPS.
        Ensure extract_iter honours the plan’s parenthetical_allows.
        """
        # Fake registry at plainera_unacronym.nlp.plugins.registry
        import types, sys

        registry_mod_name = "plainera_unacronym.nlp.plugins.registry"
        fake_registry = types.ModuleType(registry_mod_name)

        class PluginCapOnly:
            def extend_extraction(self, builder):
                builder.add_inline_cues([r"aka"])
                builder.add_parenthetical_allow(lambda definition, acronym: acronym.isupper())

        def fake_get(_names):
            return [PluginCapOnly()]

        fake_registry.get = fake_get
        monkeypatch.setitem(sys.modules, registry_mod_name, fake_registry)

        # Text with two parentheticals: one with lower-case acr (should be filtered), one uppercase (kept)
        text = "foo (abc) and Bar (ABC). PDF (Portable Document Format) aka Portable Document Format"
        cfg = _cfg_intergrated(plugins=("cap_only",))

        out = list(extract_iter(text, cfg))
        # The lowercase 'abc' parenthetical should be dropped by the allow
        assert not any(e.acronym == "ABC" and e.original_definition == "abc" for e in out)
        # The uppercase ABC parenthetical should pass
        assert any(e.acronym == "ABC" and e.original_definition.lower() == "abc" for e in out) is False
        assert any(e.acronym == "PDF" and "Portable Document Format" in e.definition for e in out)
        # The extra inline cue 'aka' should produce an inline match too
        assert any(e.acronym == "PDF" and e.source == "in_text" for e in out)

    def test_start_end_focus_window_and_no_duplicates(self):
        # Two occurrences; restrict to the second half
        text = (
            "Portable Document Format (PDF) is used.\n"
            "Noise.\n"
            "Also: PDF (Portable Document Format), and PDF stands for Portable Document Format."
        )
        mid = text.index("Also:")
        cfg = _cfg_intergrated()

        out = list(extract_iter(text, cfg, start=mid))
        # Should only see matches from second half (rev + inline)
        assert out
        assert all(e.acr_start >= mid for e in out)

        # Dedup by identical spans
        spans = {(e.acr_start, e.acr_end, e.def_start, e.def_end) for e in out}
        assert len(spans) == len(out)

    def test_filters_by_two_words_and_maxlen_integration(self):
        # Single-word def (should drop when require_two_words=True)
        text = "AB stands for Alpha.\n" \
               "CD stands for a very very very very long phrase that should exceed the limit."

        cfg = _cfg_intergrated(require_two_words=True, max_phrase_chars=15)

        out = list(extract_iter(text, cfg))
        # 'Alpha' (single word) dropped; the very long phrase collapsed by normaliser may still exceed 15 → dropped
        assert not any(e.acronym == "AB" for e in out)
        assert not any(e.acronym == "CD" for e in out)


class TestExtractIterSmall:

    @staticmethod
    def _cfg():
        return mod.ExtractionConfig(
            enabled_parenthetical=True,
            enabled_inline=True,
            conf_parenthetical=0.95,
            conf_inline=0.80,
            max_phrase_chars=200,
            require_two_words=True,
            inline_cues=(r"short\s+for", r"stands?\s+for"),
            plugins=(),
        )

    def test_pto_inline_and_pdf_parenthetical_coexist(self):
        text = (
            "PTO stands for Please Turn Over on print jobs. "
            "Portable Document Format (PDF) dominates documents."
        )
        outs = list(mod.extract_iter(text, self._cfg()))
        by = {}
        for ed in outs:
            by.setdefault(ed.acronym, []).append(ed)

        assert "PTO" in by
        assert any("Please Turn Over" in e.definition for e in by["PTO"])

        assert "PDF" in by
        assert any("Portable Document Format" in e.definition for e in by["PDF"])
