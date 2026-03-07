# needed to 'activate' plugin
import plainera_unacronym.nlp.detection.domains  # noqa: F401
from plainera_unacronym.nlp import Detector

from plainera_unacronym.nlp.common.types import DetectorConfig
from plainera_unacronym.nlp.detection.heuristics.core import compile_pattern, iter_candidates_with
from plainera_unacronym.nlp.plugins.activation import autodetect_domains
from plainera_unacronym.nlp.plugins.registry import DOMAIN_PLUGINS


class TestAutodetectDomains:

    def test_legal_plugin_registered(self):
        assert "legal" in DOMAIN_PLUGINS


    def test_autodetect_domains_enables_legal_for_means_pattern(self):
        cfg = DetectorConfig()
        text = 'In this Agreement, "Services" shall mean the services described in Schedule A.'
        auto = autodetect_domains(text, cfg)
        assert "legal" in auto


    def test_autodetect_domains_enables_legal_for_agreement_cue(self):
        cfg = DetectorConfig()
        text = "This Agreement is made on the Effective Date between the parties."
        auto = autodetect_domains(text, cfg)
        assert "legal" in auto


    def test_autodetect_domains_does_not_enable_legal_for_normal_text(self):
        cfg = DetectorConfig()
        text = "I went to the shop today. The weather was fine and nothing shall mean anything."
        auto = autodetect_domains(text, cfg)
        assert "legal" not in auto

    def test_autodetect_domains_does_not_enable_legal_for_bare_shall_mean_phrase(self):
        cfg = DetectorConfig()
        text = "I went to the shop today; nothing shall mean anything."
        auto = autodetect_domains(text, cfg)
        assert "legal" not in auto

    def test_autodetect_domains_enables_legal_for_quoted_definition(self):
        cfg = DetectorConfig()
        text = 'In this Agreement, "Services" shall mean the services described in Schedule A.'
        auto = autodetect_domains(text, cfg)
        assert "legal" in auto

    def test_detector_merges_auto_domains_into_enabled_domains(self):
        cfg = DetectorConfig(enabled_domains=frozenset())
        d = Detector(config=cfg)

        text = 'In this Agreement, "Services" means the services described in Schedule A.'
        cfg2 = d._with_auto_domains(text)

        assert "legal" in cfg2.enabled_domains

    def test_iter_candidates_with_legal_enabled_smoke(self):
        cfg = DetectorConfig(enabled_domains=frozenset({"legal"}))
        pat = compile_pattern(cfg)

        text = 'This Agreement ("Agreement") is made on the Effective Date.'
        out = list(iter_candidates_with(text, cfg, pat))

        # We don't care what it finds here; only that it runs.
        assert isinstance(out, list)
