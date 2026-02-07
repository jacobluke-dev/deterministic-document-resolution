# tests/test_plainera_unacronym/test_nlp/detection/domains/bio/test_rules.py

import re
from dataclasses import replace

import pytest

from plainera_unacronym.nlp.detection.domains.bio import rules


@pytest.fixture
def cfg() -> rules.BioConfig:
    # Keep config small + explicit so tests don’t accidentally depend on defaults.
    return rules.BioConfig(
        rna_like=frozenset({"mRNA", "miRNA", "sgRNA"}),
        two_letter_keep=frozenset({"OR", "HR", "RR"}),
        stats_window_chars=60,
    )


class TestExtraCandidates:
    def test_yields_regex_matches_from_bio_pattern(self, monkeypatch, cfg):
        # Make the bio pattern deterministic for this unit.
        pat = re.compile(r"(?P<bio>IL-\d{1,3}|SARS-CoV-2)")
        monkeypatch.setattr(rules, "bio_pattern", lambda: pat, raising=True)

        text = "Measured IL-6 and SARS-CoV-2 in samples."
        hits = list(rules.extra_candidates(text, cfg))

        assert ("IL-6", text.index("IL-6"), text.index("IL-6") + 4) in hits
        assert ("SARS-CoV-2", text.index("SARS-CoV-2"), text.index("SARS-CoV-2") + 10) in hits
        assert all(text[s:e] == surf for surf, s, e in hits)

    def test_adds_explicit_rna_like_tokens(self, monkeypatch, cfg):
        # Pattern yields nothing; RNA additions should still appear.
        monkeypatch.setattr(rules, "bio_pattern", lambda: re.compile(r"(?P<bio>NO_MATCH)"), raising=True)

        text = "We quantified mRNA and miRNA; sgRNA also appeared."
        hits = list(rules.extra_candidates(text, cfg))
        surfaces = [s for s, _, _ in hits]

        assert "mRNA" in surfaces
        assert "miRNA" in surfaces
        assert "sgRNA" in surfaces

    def test_rna_like_disabled_does_not_add_extra(self, monkeypatch, cfg):
        monkeypatch.setattr(rules, "bio_pattern", lambda: re.compile(r"(?P<bio>NO_MATCH)"), raising=True)
        cfg2 = replace(cfg, rna_like=frozenset())

        text = "We quantified mRNA and miRNA."
        hits = list(rules.extra_candidates(text, cfg2))

        assert hits == []

    def test_rna_like_word_boundary_excludes_nonword_suffix(self, monkeypatch, cfg):

        cfg2 = replace(cfg, rna_like=frozenset({"mRNA+", "miRNA"}))
        monkeypatch.setattr(rules, "bio_pattern", lambda: re.compile(r"(?P<bio>NO_MATCH)"), raising=True)

        text = "Signals: mRNA+ and miRNA."
        hits = list(rules.extra_candidates(text, cfg2))
        surfaces = [s for s, _, _ in hits]

        assert "miRNA" in surfaces
        assert "mRNA+" not in surfaces


class TestSentenceSlice:
    def test_bounds_to_sentence_terminators(self):
        text = "Alpha one. Beta two? Gamma three! Delta."
        s = text.index("Gamma")
        e = s + len("Gamma")

        a, b = rules._sentence_slice(text, s, e, max_chars=10_000)

        # Should slice the full "sentence" between '?' and '!' (excluding '?', excluding '!')
        assert text[a:b] == " Gamma three"
        assert a == text.index(" Gamma three")
        assert b == text.index("!")

    def test_no_terminators_falls_back_to_text_edges(self):
        text = "no terminators anywhere"
        s = text.index("terminators")
        e = s + len("terminators")

        a, b = rules._sentence_slice(text, s, e, max_chars=10_000)
        assert (a, b) == (0, len(text))

    def test_soft_clamp_applies_on_long_sentence(self):
        text = "A" * 500
        s, e = 200, 210

        a, b = rules._sentence_slice(text, s, e, max_chars=51)

        assert b - a <= 51
        mid = (s + e) // 2
        assert a <= mid <= b


class TestBioKeepGuard:
    def test_keeps_rna_like_unconditionally(self, cfg):
        text = "Observed miRNA in sample."
        s = text.index("miRNA")
        e = s + 4

        assert rules.bio_keep_guard("miRNA", text, s, e, cfg) is True

    def test_keeps_two_letter_only_with_stats_context(self, cfg):
        text = "OR = 1.8 (95% CI 1.2–2.3) for treatment."
        s = text.index("OR")
        e = s + 2

        assert rules.bio_keep_guard("OR", text, s, e, cfg) is True

    def test_does_not_keep_two_letter_without_stats_context(self, cfg):
        text = "OR of many options were discussed casually."
        s = text.index("OR")
        e = s + 2

        assert rules.bio_keep_guard("OR", text, s, e, cfg) is False

    def test_requires_two_letter_uppercase(self, cfg):
        text = "Or = 1.8 (95% CI 1.2–2.3) for treatment."
        s = text.index("Or")
        e = s + 2

        assert rules.bio_keep_guard("Or", text, s, e, cfg) is False

    def test_respects_two_letter_keep_list(self, cfg):
        text = "OK = 1.8 (95% CI 1.2–2.3) for treatment."
        s = text.index("OK")
        e = s + 2

        assert rules.bio_keep_guard("OK", text, s, e, cfg) is False

    def test_uses_sentence_slice_window(self, monkeypatch, cfg):
        # Prove we only scan a bounded window by shrinking the window and placing stats outside it.
        cfg2 = replace(cfg, stats_window_chars=10)

        text = "OR blahblahblahblah (95% CI 1.2–2.3)"
        s = text.index("OR")
        e = s + 2

        # Keep rules as-is; just ensure slice is tiny and misses CI.
        assert rules.bio_keep_guard("OR", text, s, e, cfg2) is False
