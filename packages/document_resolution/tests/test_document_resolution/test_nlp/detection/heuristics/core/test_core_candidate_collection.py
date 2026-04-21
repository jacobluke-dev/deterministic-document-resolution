import re

import document_resolution.nlp.detection.heuristics.core as core
import document_resolution.nlp.plugins.registry as domain_mod
import pytest
from document_resolution.nlp.common.types import AcronymDetectorConfig
from document_resolution.nlp.detection.heuristics.core import (
    _collect_core_hits,
    _collect_domain_hits,
    iter_acronym_candidates,
)


class TestIterCandidatesWith:
    # Token pattern for tests:
    # - Captures a "token" as letters followed by letters/digits or common separators.
    # - Includes '.' so we can verify trailing-punct trimming (e.g., "RAM.")
    PAT = re.compile(r"(?P<tok>[A-Za-z][A-Za-z0-9&\-/\.]*)")

    @staticmethod
    def collect(text: str, cfg: AcronymDetectorConfig, pat: re.Pattern[str]):
        return list(iter_acronym_candidates(text, cfg, pat))

    def test_all_caps_simple(self):
        cfg = AcronymDetectorConfig()
        text = "We ran it on the GPU and CPU yesterday."
        out = self.collect(text, cfg, self.PAT)
        surfaces = [s for s, _, _ in out]
        assert "GPU" in surfaces
        assert "CPU" in surfaces

    def test_mixed_case_relaxation_enabled(self):
        # "iOS": 2/3 letters uppercase ≈ 0.667. With mixed-case relaxation (0.5) it should pass.
        cfg = AcronymDetectorConfig(enable_mixed_case=True, require_caps_ratio_mixed=0.5)
        text = "We ship an iOS build every week."
        out = self.collect(text, cfg, self.PAT)
        surfaces = [s for s, _, _ in out]
        assert "iOS" in surfaces  # relaxed threshold applied (upp >= 2)

    def test_mixed_case_relaxation_disabled(self):
        # With relaxation OFF, require_caps_ratio=0.7 and iOS has ~0.667 → should be filtered out.
        cfg = AcronymDetectorConfig(enable_mixed_case=False)
        text = "We ship an iOS build every week."
        out = self.collect(text, cfg, self.PAT)
        surfaces = [s for s, _, _ in out]
        assert "iOS" not in surfaces

    def test_digits_ignored_in_caps_ratio(self):
        # "H2O": letters H,O are uppercase; digit '2' ignored → ratio = 1.0 → passes.
        cfg = AcronymDetectorConfig()
        text = "Check the H2O level."
        out = self.collect(text, cfg, self.PAT)
        surfaces = [s for s, _, _ in out]
        assert "H2O" in surfaces

    def test_trailing_punct_stripped_and_indices(self):
        cfg = AcronymDetectorConfig()
        text = "Memory uses RAM."
        out = self.collect(text, cfg, self.PAT)
        # Expect one candidate "RAM" with indices pointing exactly to 'RAM' (not the '.')
        assert any(s == "RAM" for s, _, _ in out)
        # Verify indices are tight to the token (no trailing '.')
        for srf, s, e in out:
            if srf == "RAM":
                assert text[s:e] == "RAM"
                # e should be the position right after 'M'
                assert text[e : e + 1] == "."  # the '.' is outside the candidate

    def test_min_len_enforced(self):
        # Default min_len=2 → single-letter tokens should be filtered.
        cfg = AcronymDetectorConfig()
        text = "A B CD"
        out = self.collect(text, cfg, self.PAT)
        surfaces = [s for s, _, _ in out]
        assert "A" not in surfaces
        assert "B" not in surfaces
        assert "CD" in surfaces

    def test_max_len_enforced(self):
        # Default max_len=10 → very long all-caps should be filtered.
        cfg = AcronymDetectorConfig()
        long_tok = "THISISVERYLONG"  # length 14
        text = f"Edge {long_tok} token."
        out = self.collect(text, cfg, self.PAT)
        surfaces = [s for s, _, _ in out]
        assert long_tok not in surfaces

    def test_allowed_separators_compound_tokens(self):
        # Ensure tokens with separators (&, -) get considered and pass.
        cfg = AcronymDetectorConfig()
        text = "Our R&D team ported GPU-CPU pipelines."
        out = self.collect(text, cfg, self.PAT)
        surfaces = [s for s, _, _ in out]
        assert "R&D" in surfaces
        assert "GPU-CPU" in surfaces

    def test_mixed_case_requires_two_uppers_for_relax(self):
        # Relaxation only kicks in if upp >= 2. "eBay" has only 1 uppercase in practice (B),
        # so it should fail under default require_caps_ratio=0.7.
        cfg = AcronymDetectorConfig(enable_mixed_case=False)
        text = "We listed it on eBay."
        out = self.collect(text, cfg, self.PAT)
        surfaces = [s for s, _, _ in out]
        assert "eBay" not in surfaces

    def test_mixed_case_relax_threshold_param(self):
        # Tighten the mixed-case threshold so "NaCl" (2/4 = 0.5) fails.
        cfg = AcronymDetectorConfig(enable_mixed_case=True, require_caps_ratio_mixed=0.6)
        text = "We used NaCl in the experiment."
        out = self.collect(text, cfg, self.PAT)
        surfaces = [s for s, _, _ in out]
        assert "NaCl" not in surfaces


class TestCollectDomainHits:
    def test_collects_and_sorts_by_start_then_length_desc(self, monkeypatch):
        text = "The ABC transporter and ABCDEF domain are here."

        # Fake plugins
        class BioPlug:
            def extra_candidates(self, _text, _cfg):
                # Same start 10, different lengths; plus another span at start 5
                return [
                    ("bio", 10, 15),  # len 5
                    ("bio", 10, 20),  # len 10 (should come before len 5)
                    ("bio", 5, 7),  # earlier start (should be first overall)
                ]

        class FinPlug:
            def extra_candidates(self, _text, _cfg):
                return [("fin", 12, 18)]  # middle

        monkeypatch.setattr(core, "DOMAIN_PLUGINS", {"bio": BioPlug(), "finance": FinPlug()}, raising=False)

        # Accept everything by echoing a Span-like tuple
        def accept(text_arg, cfg_arg, s, e):
            return ("hit", s, e)

        monkeypatch.setattr(core, "_accept_candidate", accept, raising=False)

        cfg = AcronymDetectorConfig(enabled_domains=("bio", "finance"))
        hits = _collect_domain_hits(text, cfg)

        # Expect sort: start asc (5..7), then 10..20 (longer first), then 10..15, then 12..18
        assert hits == [
            ("hit", 5, 7),
            ("hit", 10, 20),
            ("hit", 10, 15),
            ("hit", 12, 18),
        ]

    def test_filters_by_enabled_domains_only(self, monkeypatch):
        class BioPlug:
            def extra_candidates(self, *_):
                return [("bio", 1, 3)]

        class FinPlug:
            def extra_candidates(self, *_):
                return [("fin", 100, 110)]

        monkeypatch.setattr(core, "DOMAIN_PLUGINS", {"bio": BioPlug(), "finance": FinPlug()}, raising=False)
        monkeypatch.setattr(core, "_accept_candidate", lambda _t, _c, s, e: ("hit", s, e), raising=False)

        cfg = AcronymDetectorConfig(enabled_domains=frozenset({"bio"}))  # finance disabled
        hits = _collect_domain_hits("x", cfg)
        assert hits == [("hit", 1, 3)]

    def test_skips_missing_plugins_and_rejections_and_none_returns(self, monkeypatch):
        class BioPlug:
            def extra_candidates(self, *_):
                return [("bio", 5, 9), ("bio", 50, 60)]

        monkeypatch.setattr(core, "DOMAIN_PLUGINS", {"bio": BioPlug()}, raising=False)

        def accept(_t, _c, s, e):
            return None if (s, e) == (50, 60) else ("hit", s, e)

        monkeypatch.setattr(core, "_accept_candidate", accept, raising=False)

        cfg = AcronymDetectorConfig(enabled_domains=frozenset({"bio", "chem"}))  # "chem" missing -> ignored
        hits = _collect_domain_hits("x", cfg)
        assert hits == [("hit", 5, 9)]

    def test_empty_enabled_domains_or_none_yields_no_hits(self, monkeypatch):
        # Even if there are plugins, with enabled_domains empty/None the loop is skipped.
        class AnyPlug:
            def extra_candidates(self, *_):
                return [("x", 1, 2)]

        monkeypatch.setattr(domain_mod, "DOMAIN_PLUGINS", {"any": AnyPlug()}, raising=False)
        monkeypatch.setattr(domain_mod, "_accept_candidate", lambda *_: ("hit", 1, 2), raising=False)

        assert _collect_domain_hits("x", AcronymDetectorConfig(enabled_domains=(frozenset()))) == []
        assert _collect_domain_hits("x", AcronymDetectorConfig(enabled_domains=None)) == []


class TestCollectCoreHits:
    def test_collects_in_text_order(self, monkeypatch):
        # Pattern: named group 'tok' for ALL-CAPS tokens length>=2
        pat = re.compile(r"(?P<tok>[A-Z]{2,})")

        text = "xx ABC yy DEF and GHIJ."
        # Echo back a Span-like tuple to simulate acceptance
        monkeypatch.setattr(core, "_accept_candidate", lambda _t, _c, s, e: ("hit", s, e), raising=False)

        # Minimal config object (fields unused by our stub)
        class DetectorConfig: ...

        cfg = DetectorConfig()

        hits = _collect_core_hits(text, cfg, pat)
        # Expect left-to-right order by match positions
        assert hits == [
            ("hit", text.index("ABC"), text.index("ABC") + 3),
            ("hit", text.index("DEF"), text.index("DEF") + 3),
            ("hit", text.index("GHIJ"), text.index("GHIJ") + 4),
        ]

    def test_rejected_hits_are_filtered_out(self, monkeypatch):
        pat = re.compile(r"(?P<tok>[A-Z]{2,})")
        text = "ABC DEF GHI"

        def accept(_t, _c, s, e):
            # Reject the middle token DEF
            tok = text[s:e]
            return None if tok == "DEF" else ("hit", s, e)

        monkeypatch.setattr(core, "_accept_candidate", accept, raising=False)

        class DetectorConfig: ...

        cfg = DetectorConfig()

        hits = _collect_core_hits(text, cfg, pat)
        assert [text[s:e] for (_, s, e) in hits] == ["ABC", "GHI"]

    def test_no_matches_returns_empty(self, monkeypatch):
        pat = re.compile(r"(?P<tok>[A-Z]{2,})")
        text = "no caps here"

        monkeypatch.setattr(core, "_accept_candidate", lambda *_: ("hit", 0, 0), raising=False)

        class DetectorConfig: ...

        cfg = DetectorConfig()

        assert _collect_core_hits(text, cfg, pat) == []

    def test_requires_named_group_tok(self, monkeypatch):
        # Pattern without 'tok' should raise (m.span('tok') IndexError)
        pat = re.compile(r"([A-Z]{2,})")
        text = "ABC"

        monkeypatch.setattr(core, "_accept_candidate", lambda *_: ("hit", 0, 3), raising=False)

        class DetectorConfig: ...

        cfg = DetectorConfig()

        with pytest.raises(IndexError):
            _collect_core_hits(text, cfg, pat)



