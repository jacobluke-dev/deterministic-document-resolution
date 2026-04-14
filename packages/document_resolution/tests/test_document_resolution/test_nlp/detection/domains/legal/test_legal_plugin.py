# needed to 'activate' plugin
import document_resolution.nlp.detection.domains  # noqa: F401
from document_resolution.nlp.common.types import AcronymDetectorConfig
from document_resolution.nlp.detection.acronym.compiler import compile_acronym_pattern
from document_resolution.nlp.detection.acronym.detector import AcronymDetector
from document_resolution.nlp.detection.domains import LegalPlugin
from document_resolution.nlp.detection.domains.legal.config import (
    LEGAL_CONTRACT_SUBJECT_SHALL_RE,
    LEGAL_QUOTED_MEANS_RE,
)
from document_resolution.nlp.detection.domains.legal.legal_gate import should_enable_legal
from document_resolution.nlp.detection.heuristics.core import iter_acronym_candidates
from document_resolution.nlp.plugins.activation import autodetect_domains
from document_resolution.nlp.plugins.registry import DOMAIN_PLUGINS


class TestAutodetectDomains:

    def test_legal_plugin_registered(self):
        assert "legal" in DOMAIN_PLUGINS


    def test_autodetect_domains_enables_legal_for_means_pattern(self):
        cfg = AcronymDetectorConfig()
        text = 'In this Agreement, "Services" shall mean the services described in Schedule A.'
        auto = autodetect_domains(text, cfg)
        assert "legal" in auto


    def test_autodetect_domains_enables_legal_for_agreement_cue(self):
        cfg = AcronymDetectorConfig()
        text = "This Agreement is made on the Effective Date between the parties."
        auto = autodetect_domains(text, cfg)
        assert "legal" in auto

    def test_autodetect_domains_does_not_enable_legal_for_technical_doc_structure(self):
        cfg = AcronymDetectorConfig()
        text = "Section 2 describes the architecture. Appendix A lists components."
        auto = autodetect_domains(text, cfg)
        assert "legal" not in auto


    def test_autodetect_domains_does_not_enable_legal_for_normal_text(self):
        cfg = AcronymDetectorConfig()
        text = "I went to the shop today. The weather was fine and nothing shall mean anything."
        auto = autodetect_domains(text, cfg)
        assert "legal" not in auto

    def test_autodetect_domains_enables_legal_for_quoted_definition(self):
        cfg = AcronymDetectorConfig()
        text = 'In this Agreement, "Services" shall mean the services described in Schedule A.'
        auto = autodetect_domains(text, cfg)
        assert "legal" in auto

    def test_detector_merges_auto_domains_into_enabled_domains(self):
        cfg = AcronymDetectorConfig(enabled_domains=frozenset())
        d = AcronymDetector(config=cfg)

        text = 'In this Agreement, "Services" means the services described in Schedule A.'
        cfg2 = d._with_auto_domains(text)

        assert "legal" in cfg2.enabled_domains

    def test_iter_candidates_with_legal_enabled_smoke(self):
        cfg = AcronymDetectorConfig(enabled_domains=frozenset({"legal"}))
        pat = compile_acronym_pattern(cfg)

        text = 'This Agreement ("Agreement") is made on the Effective Date.'
        out = list(iter_acronym_candidates(text, cfg, pat))

        # We don't care what it finds here; only that it runs.
        assert isinstance(out, list)


class TestLegalGate:
    def test_enables_on_strong_quoted_definition(self):
        ok, reasons = should_enable_legal(
            'In this Agreement, "Services" shall mean the services described in Schedule A.')
        assert ok is True
        assert "quoted_means" in reasons

    def test_enables_on_strong_hereinafter(self):
        ok, reasons = should_enable_legal("Acme Ltd (hereinafter the Supplier) agrees as follows.")
        assert ok is True
        assert "hereinafter" in reasons

    def test_does_not_enable_on_bare_shall_mean_phrase(self):
        ok, reasons = should_enable_legal("Nothing shall mean anything to anyone.")
        assert ok is False
        assert reasons == []

    def test_does_not_enable_on_technical_structure_only(self):
        ok, reasons = should_enable_legal("Section 2 describes the architecture. Appendix A lists components.")
        assert ok is False

    def test_enables_on_quoted_definition(self):
        ok, reasons = should_enable_legal(
            'In this Agreement, "Services" shall mean the services described in Schedule A.')
        assert ok is True
        assert "quoted_means" in reasons

    def test_enables_on_hereinafter(self):
        ok, reasons = should_enable_legal("Acme Ltd (hereinafter the Supplier) agrees as follows.")
        assert ok is True
        assert "hereinafter" in reasons

    def test_should_enable_legal_for_quoted_means(self):
        text = '"Agreement" means this contract.'
        ok, reasons = should_enable_legal(text)
        assert ok is True
        assert "quoted_means" in reasons

    def test_should_enable_legal_for_contractual_shall_language(self):
        text = 'The Services shall commence on Monday.'
        ok, reasons = should_enable_legal(text)
        assert ok is True

    def test_should_enable_legal_for_combined_contract_signals(self):
        text = """
        "Agreement" means this contract.
        "Services" means the software development services.
        The Services shall commence on Monday.
        """.strip()
        ok, reasons = should_enable_legal(text)
        assert ok is True

    def test_quoted_means_is_detected(self):
        assert LEGAL_QUOTED_MEANS_RE.search('"Agreement" means this contract.')

    def test_contract_subject_shall_is_detected(self):
        assert LEGAL_CONTRACT_SUBJECT_SHALL_RE.search("The Services shall commence on Monday.")

    def test_bare_shall_phrase_is_not_detected(self):
        assert not LEGAL_CONTRACT_SUBJECT_SHALL_RE.search("Nothing shall mean anything to anyone.")

    def test_should_enable_legal_for_contract_defined_terms(self):
        text = """
        "Agreement" means this contract.
        "Services" means the software development, support, and maintenance services.
        The Services shall commence on Monday.
        """.strip()

        ok, reasons = should_enable_legal(text)
        assert ok is True

class TestLegalExtraCandidates:
    def test_extra_candidates_disabled_when_domain_not_enabled(self):
        plug = LegalPlugin()
        cfg = AcronymDetectorConfig(enabled_domains=frozenset())
        text = "Regulation (EU) 2016/679 applies."
        assert list(plug.extra_candidates(text, cfg) or []) == []

    def test_extra_candidates_emits_when_enabled(self):
        plug = LegalPlugin()
        cfg = AcronymDetectorConfig(enabled_domains=frozenset({"legal"}))
        text = "Regulation (EU) 2016/679 applies."
        out = list(plug.extra_candidates(text, cfg) or [])
        assert out  # at least one
