# needed to 'activate' plugin
import plainera_unacronym.nlp.detection.domains  # noqa: F401
from plainera_unacronym.nlp import AcronymDetector
from plainera_unacronym.nlp.common.types import AcronymDetectorConfig
from plainera_unacronym.nlp.detection.acronym.compiler import compile_acronym_pattern
from plainera_unacronym.nlp.detection.domains import StructuralReferencePlugin
from plainera_unacronym.nlp.detection.domains.structural_reference.config import (
    STRUCT_APPENDIX_RE,
    STRUCT_CLAUSE_RE,
    STRUCT_SCHEDULE_RE,
    STRUCT_SECTION_RE,
)
from plainera_unacronym.nlp.detection.domains.structural_reference.structural_gate import (
    should_enable_structural_reference,
)
from plainera_unacronym.nlp.detection.heuristics.core import iter_acronym_candidates
from plainera_unacronym.nlp.plugins.activation import autodetect_domains
from plainera_unacronym.nlp.plugins.registry import DOMAIN_PLUGINS


class TestAutodetectDomains:
    def test_structural_reference_plugin_registered(self):
        assert "structural_reference" in DOMAIN_PLUGINS

    def test_autodetect_domains_enables_structural_reference_for_schedule_reference(self):
        cfg = AcronymDetectorConfig()
        text = "The obligations are set out in Schedule A."
        auto = autodetect_domains(text, cfg)
        assert "structural_reference" in auto

    def test_autodetect_domains_enables_structural_reference_for_repeated_section_references(self):
        cfg = AcronymDetectorConfig()
        text = "Section 1 describes the system. Section 2 defines the interfaces."
        auto = autodetect_domains(text, cfg)
        assert "structural_reference" in auto

    def test_autodetect_domains_enables_structural_reference_for_mixed_weak_signals(self):
        cfg = AcronymDetectorConfig()
        text = "Clause 2.1 sets out the process. Section 4 contains the exceptions."
        auto = autodetect_domains(text, cfg)
        assert "structural_reference" in auto

    def test_autodetect_domains_does_not_enable_structural_reference_for_normal_text(self):
        cfg = AcronymDetectorConfig()
        text = "I went to the shop today. The weather was fine and the team discussed the plan."
        auto = autodetect_domains(text, cfg)
        assert "structural_reference" not in auto

    def test_autodetect_domains_does_not_enable_structural_reference_for_single_weak_reference(self):
        cfg = AcronymDetectorConfig()
        text = "Section 2 describes the architecture."
        auto = autodetect_domains(text, cfg)
        assert "structural_reference" not in auto

    def test_detector_merges_auto_domains_into_enabled_domains(self):
        cfg = AcronymDetectorConfig(enabled_domains=frozenset())
        d = AcronymDetector(config=cfg)

        text = "The obligations are set out in Schedule A."
        cfg2 = d._with_auto_domains(text)

        assert "structural_reference" in cfg2.enabled_domains

    def test_iter_candidates_with_structural_reference_enabled_smoke(self):
        cfg = AcronymDetectorConfig(enabled_domains=frozenset({"structural_reference"}))
        pat = compile_acronym_pattern(cfg)

        text = "Section 1 defines the API."
        out = list(iter_acronym_candidates(text, cfg, pat))

        # We do not care what it finds here; only that it runs unchanged.
        assert isinstance(out, list)


class TestStructuralGate:
    def test_enables_on_strong_schedule_reference(self):
        ok, reasons = should_enable_structural_reference(
            "The obligations are set out in Schedule A."
        )
        assert ok is True
        assert "schedule" in reasons

    def test_enables_on_strong_appendix_reference(self):
        ok, reasons = should_enable_structural_reference(
            "Appendix B contains the supporting tables."
        )
        assert ok is True
        assert "appendix" in reasons

    def test_enables_on_repeated_section_references(self):
        ok, reasons = should_enable_structural_reference(
            "Section 1 describes the system. Section 2 defines the interfaces."
        )
        assert ok is True
        assert "section" in reasons
        assert "repeated_section" in reasons

    def test_enables_on_repeated_clause_references(self):
        ok, reasons = should_enable_structural_reference(
            "Clause 2 applies. Clause 2.1 sets out the detailed exceptions."
        )
        assert ok is True
        assert "clause" in reasons
        assert "repeated_clause" in reasons

    def test_enables_on_mixed_weak_signals(self):
        ok, reasons = should_enable_structural_reference(
            "Clause 2.1 sets out the process. Section 4 contains the exceptions."
        )
        assert ok is True
        assert "clause" in reasons
        assert "section" in reasons

    def test_does_not_enable_on_single_weak_section_reference(self):
        ok, reasons = should_enable_structural_reference(
            "Section 2 describes the architecture."
        )
        assert ok is False
        assert "section" in reasons



    def test_does_not_enable_on_normal_text(self):
        ok, reasons = should_enable_structural_reference(
            "I went to the shop today. The weather was fine and the team discussed the plan."
        )
        assert ok is False
        assert reasons == []

    def test_schedule_regex_is_detected(self):
        assert STRUCT_SCHEDULE_RE.search("Schedule A contains the deliverables.")

    def test_appendix_regex_is_detected(self):
        assert STRUCT_APPENDIX_RE.search("Appendix B contains the supporting tables.")

    def test_section_regex_is_detected(self):
        assert STRUCT_SECTION_RE.search("Section 4.2 describes the interface.")

    def test_clause_regex_is_detected(self):
        assert STRUCT_CLAUSE_RE.search("Clause 7.2 applies to payment terms.")


class TestStructuralReferencePlugin:
    def test_extra_candidates_disabled_when_domain_not_enabled(self):
        plug = StructuralReferencePlugin()
        cfg = AcronymDetectorConfig(enabled_domains=frozenset())
        text = "Schedule A contains the deliverables."
        assert list(plug.extra_candidates(text, cfg) or []) == []

    def test_extra_candidates_still_empty_when_enabled(self):
        plug = StructuralReferencePlugin()
        cfg = AcronymDetectorConfig(enabled_domains=frozenset({"structural_reference"}))
        text = "Schedule A contains the deliverables."
        assert list(plug.extra_candidates(text, cfg) or []) == []

    def test_keep_guard_always_false(self):
        plug = StructuralReferencePlugin()
        cfg = AcronymDetectorConfig(enabled_domains=frozenset({"structural_reference"}))
        text = "Section 1 defines the API."
        assert plug.keep_guard("API", text, 22, 25, cfg) is False
