import plainera_unacronym.nlp.detection.detector as det
import plainera_unacronym.nlp.plugins.activation as act
import pytest
from plainera_unacronym.nlp.detection.detector import Detector, DetectorConfig, autodetect_domains
from plainera_unacronym.nlp.detection.domains.bio.config import BioConfig
from plainera_unacronym.nlp.detection.domains.bio.plugin import BioPlugin


class TestBioAutodetect:
    def test_autodetect_domains_flags_bio_from_rna_and_cytokines(self, monkeypatch):
        """
        Integration: ensure autodetect_domains returns {'bio'} when the text contains
        strong bio signals (e.g., mRNA / IL-6 / SARS-CoV-2).
        """
        # The autodetect uses an isinstance(plug, SupportsSniff) gate and a safe wrapper.
        # Make sure the registered BioPlugin gets queried in this test:
        monkeypatch.setattr(act, "SupportsSniff", object, raising=True)  # all objects pass isinstance
        # Some versions implement a sandbox wrapper; route to plugin.sniff in a safe way.
        monkeypatch.setattr(act, "_safe_sniff", lambda plug, t: plug.sniff(t), raising=True)

        text = "We quantified mRNA for IL-6 after SARS-CoV-2 infection. " "The 5′UTR also showed changes."
        cfg = DetectorConfig()
        auto = autodetect_domains(text, cfg)
        assert "bio" in auto, f"Expected 'bio' in autodetected domains, got {auto}"

    def test_detector_merges_auto_domains_and_logs_added(self, patch_sink_and_logger, monkeypatch):
        # Force auto-detection to return {'bio'} where Detector will actually look:
        monkeypatch.setattr(det, "autodetect_domains", lambda text, cfg: frozenset({"bio"}), raising=True)

        # Keep detection path minimal
        monkeypatch.setattr(det, "compile_pattern", lambda _cfg: object(), raising=True)
        monkeypatch.setattr(det, "iter_candidates_with", lambda *a, **k: [], raising=True)

        d = Detector(DetectorConfig(enabled_domains=frozenset()))
        _ = d.detect("mRNA and IL-6 were measured.")

        messages = [c["message"] for c in patch_sink_and_logger]

        assert "detector.autodetect_domains" in messages, f"logs: {messages}"
        assert "detector.detect.start" in messages
        assert "detector.detect.summary" in messages
        assert messages.index("detector.autodetect_domains") < messages.index("detector.detect.start")

    def test_bio_keep_guard_for_rna_and_two_letter_stats(self):
        """
        Check BioPlugin.keep_guard behavior:
          - RNA-like surfaces are kept.
          - Two-letter tokens like OR are kept when a stats context (95% CI / OR/HR/RR) is nearby.
        """
        plug = BioPlugin()
        cfg = DetectorConfig(
            enabled_domains=frozenset({"bio"}),  # activate plugin behavior
        )

        # 1) RNA-like surface is kept unconditionally when domain is enabled.
        text1 = "Differential expression of miRNA was observed."
        s1 = text1.index("miRNA")
        e1 = s1 + len("miRNA")
        assert plug.keep_guard("miRNA", text1, s1, e1, cfg) is True

        # 2) Two-letter stat token kept only with stats context (95% CI / OR/HR/RR around it)
        text2 = "OR = 1.8 (95% CI 1.2–2.3) for the treatment group."
        s2 = text2.index("OR")
        e2 = s2 + 2
        assert plug.keep_guard("OR", text2, s2, e2, cfg) is True

        text3 = "OR of many options were discussed casually (no stats)."
        s3 = text3.index("OR")
        e3 = s3 + 2
        # Without stats context, two-letter keep should be False
        assert plug.keep_guard("OR", text3, s3, e3, cfg) is False

    def test_bio_keep_guard_fails_for_two_letter_camelcase_or(self):
        """
        Check BioPlugin.keep_guard behavior:
          - RNA-like surfaces are kept.
          - Two-letter tokens like 'Or' fail rather than 'OR'.
        """
        plug = BioPlugin()
        cfg = DetectorConfig(
            enabled_domains=frozenset({"bio"}),  # activate plugin behavior
        )

        # 1) RNA-like surface is kept unconditionally when domain is enabled.
        text1 = "Differential expression of mirNA was observed."
        s1 = text1.index("mirNA")
        e1 = s1 + len("mirNA")
        assert plug.keep_guard("mirNA", text1, s1, e1, cfg) is False

        # 2) Two-letter stat token kept only with stats context (95% CI / OR/HR/RR around it)
        text2 = "Or = 1.8 (95% CI 1.2–2.3) for the treatment group."
        s2 = text2.index("Or")
        e2 = s2 + 2
        assert plug.keep_guard("OR", text2, s2, e2, cfg) is False

    def test_keep_guard_returns_false_when_domain_disabled(self):
        plug = BioPlugin()
        cfg = DetectorConfig(enabled_domains=frozenset())
        text = "Differential expression of miRNA was observed."
        s = text.index("miRNA")
        e = s + len("miRNA")
        assert plug.keep_guard("miRNA", text, s, e, cfg) is False

    def test_keep_guard_uses_domain_cfg_override(self):
        plug = BioPlugin()
        cfg = DetectorConfig(enabled_domains=frozenset({"bio"}))
        # Override to make stats window tiny so context is missed.
        object.__setattr__(
            cfg, "domain_cfg", {"bio": BioConfig(stats_window_chars=5, two_letter_keep=frozenset({"OR"}))}
        )

        text = "OR = 1.8 (95% CI 1.2–2.3)"
        s = text.index("OR")
        e = s + 2
        assert plug.keep_guard("OR", text, s, e, cfg) is False


class TestExtraCandidates:
    def test_extra_candidates_respects_enabled_domains(self):
        plug = BioPlugin()
        text = "Measured IL-6 and IFN-γ in SARS-CoV-2 samples."
        cfg_off = DetectorConfig(enabled_domains=frozenset())
        assert list(plug.extra_candidates(text, cfg_off) or []) == []

        cfg_on = DetectorConfig(enabled_domains=frozenset({"bio"}))
        hits = list(plug.extra_candidates(text, cfg_on) or [])
        assert any(s == "IL-6" for s, _, _ in hits)
        assert all(text[s:e] == surf for surf, s, e in hits)


class TestAutoDetectedDomains:
    def test_autodetect_domains_swallows_plugin_exceptions(self, monkeypatch):
        class BadPlug(BioPlugin):
            name = "bio"

            @staticmethod
            def sniff(text: str) -> bool:
                raise RuntimeError("boom")

        monkeypatch.setattr(act, "DOMAIN_PLUGINS", {"bio": BadPlug()}, raising=True)
        cfg = DetectorConfig()
        assert act.autodetect_domains("mRNA IL-6", cfg) == frozenset()

    def test_autodetect_domains_returns_plugin_name_on_true(self, monkeypatch):
        class GoodPlug:
            name = "bio"

            @staticmethod
            def sniff(text: str) -> bool:
                return "IL-6" in text

        monkeypatch.setattr(act, "DOMAIN_PLUGINS", {"bio": GoodPlug()}, raising=True)

        cfg = DetectorConfig()
        assert act.autodetect_domains("x IL-6 y", cfg) == frozenset({"bio"})
        assert act.autodetect_domains("no signals", cfg) == frozenset()
