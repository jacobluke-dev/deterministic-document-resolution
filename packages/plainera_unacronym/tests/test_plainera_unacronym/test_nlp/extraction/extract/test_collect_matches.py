import re
from types import SimpleNamespace as NS
import pytest

import plainera_unacronym.nlp.extraction.extract_first_occ as mod
from plainera_unacronym.nlp.extraction.extract import _collect_matches


def _cfg(**overrides):
    base = dict(
        enabled_parenthetical=True,
        enabled_inline=True,
        conf_parenthetical=0.95,
        conf_inline=0.80,
        max_phrase_chars=200,
        require_two_words=True,
        min_acr_len=2,
        max_acr_len=10,
        acr_allowed=r"A-Z0-9&./-",
        inline_cues=(r"short\s+for", r"stands?\s+for"),
        plugins=(),
    )
    base.update(overrides)
    return mod.ExtractionConfig(**base)


def _plan(**overrides):
    # Minimal plan object for tests
    return NS(inline_cues=overrides.get("inline_cues", ()), parenthetical_allows=overrides.get("parenthetical_allows", ()))


class TestCollectMatchesUnit:
    def test_inline_two_word_gate(self, monkeypatch):
        text = "XK, short for x"
        pat = re.compile(r"\b(?P<acr>XK)\b\s*,?\s*short\s+for\s+(?P<def>[^){}]{1,200}?)")

        cfg = _cfg(require_two_words=True)
        seen = set()
        out = list(_collect_matches(text, pat, cfg=cfg, plan=_plan(), base_conf=cfg.conf_inline,
                                        is_parenthetical=False, seen=seen, start=0, end=len(text)))
        assert out == []  # single-word def rejected

        cfg2 = _cfg(require_two_words=False)
        seen.clear()
        out2 = list(_collect_matches(text, pat, cfg=cfg2, plan=_plan(), base_conf=cfg2.conf_inline,
                                         is_parenthetical=False, seen=seen, start=0, end=len(text)))
        assert len(out2) == 1
        assert out2[0].acronym == "XK"
        assert out2[0].definition == "x"

    def test_parenthetical_allow_predicate_applied(self, monkeypatch):
        text = "Alpha Beta (ab)"
        pat = re.compile(r"\b(?P<def>Alpha Beta)\s*\(\s*(?P<acr>ab)\s*\)")
        # Deterministically disallow via the plan (plugins)
        plan = _plan(parenthetical_allows=(lambda d, a: False,))
        cfg = _cfg()

        out = list(_collect_matches(
            text, pat, cfg=cfg, plan=plan, base_conf=cfg.conf_parenthetical,
            is_parenthetical=True, seen=set(), start=0, end=len(text)
        ))
        assert out == []  # dropped by allow

        # And a positive control to confirm we do allow when the predicate returns True
        plan_ok = _plan(parenthetical_allows=(lambda d, a: "Alpha" in d,))
        out_ok = list(_collect_matches(
            text, pat, cfg=cfg, plan=plan_ok, base_conf=cfg.conf_parenthetical,
            is_parenthetical=True, seen=set(), start=0, end=len(text)
        ))
        assert len(out_ok) == 1
        assert out_ok[0].acronym == "AB"
        assert out_ok[0].definition == "Alpha Beta"

    def test_seen_deduplicates_exact_span(self):
        text = "Portable Document Format (PDF)"
        p1 = re.compile(r"\b(?P<def>Portable Document Format)\s*\(\s*(?P<acr>PDF)\s*\)")
        p2 = re.compile(r"\b(?P<def>Portable Document Format)\s*\(\s*(?P<acr>PDF)\s*\)")

        cfg = _cfg()
        plan = _plan()
        seen = set()

        out1 = list(_collect_matches(text, p1, cfg=cfg, plan=plan, base_conf=cfg.conf_parenthetical,
                                         is_parenthetical=True, seen=seen, start=0, end=len(text)))
        out2 = list(_collect_matches(text, p2, cfg=cfg, plan=plan, base_conf=cfg.conf_parenthetical,
                                         is_parenthetical=True, seen=seen, start=0, end=len(text)))
        assert len(out1) == 1
        assert out2 == []  # duplicate span suppressed

    def test_window_limits(self):
        text = "Portable Document Format (PDF) ... PDF (Portable Document Format)"
        fwd = re.compile(r"\b(?P<def>Portable Document Format)\s*\(\s*(?P<acr>PDF)\s*\)")
        rev = re.compile(r"\b(?P<acr>PDF)\s*\(\s*(?P<def>Portable Document Format)\s*\)")
        mid = text.index("...") + 3

        cfg = _cfg()
        plan = _plan()

        # Only the reverse match in the second half
        out = list(_collect_matches(text, rev, cfg=cfg, plan=plan, base_conf=cfg.conf_parenthetical,
                                        is_parenthetical=True, seen=set(), start=mid, end=len(text)))
        assert len(out) == 1
        assert out[0].acr_start >= mid

        # Forward match should not be found in the second half
        out2 = list(_collect_matches(text, fwd, cfg=cfg, plan=plan, base_conf=cfg.conf_parenthetical,
                                         is_parenthetical=True, seen=set(), start=mid, end=len(text)))
        assert out2 == []
