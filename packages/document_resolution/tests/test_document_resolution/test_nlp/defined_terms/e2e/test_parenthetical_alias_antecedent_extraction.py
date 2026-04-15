from document_resolution.nlp.common.types import DefinedTermDetectorConfig
from document_resolution.nlp.extraction.defined_terms.execute import detect_and_resolve_terms
from test_document_resolution.test_nlp.defined_terms.e2e.defined_terms_e2e_common import (
    meaning_by_id,
    resolutions_for_key,
)

"""
**antecedent extraction for parenthetical defined-term aliases**, so when the parser sees
`This Master Services Agreement (the "Agreement")`, it keeps not just the alias term `"Agreement"`
but also the thing it aliases: `This Master Services Agreement`. The goal is to preserve that richer
meaning **conservatively and traceably**, without accidentally swallowing trailing prose or breaking
existing `means` / `shall mean` definition extraction.
"""
class TestParentheticalAliasAntecedentExtraction:

    def test_parenthetical_alias_extracts_antecedent_phrase(self):
        text = """
        This Master Services Agreement (the "Agreement") is entered into on the Effective Date.
        "Effective Date" means the date on which both Parties sign this Agreement.
        """.strip()

        det_cfg = DefinedTermDetectorConfig(
            unquoted_capitalised_terms_policy="always",
        )

        det_res, extr, reports, state = detect_and_resolve_terms(
            text,
            det_cfg=det_cfg,
            return_reports=True,
            return_state=True,
        )

        intro_keys = {i.normalized_key for i in det_res.introductions}
        assert "agreement" in intro_keys

        agreement_resolutions = resolutions_for_key(extr, "agreement")
        assert len(agreement_resolutions) >= 1
        assert agreement_resolutions[0].chosen_meaning_id is not None

        meaning = meaning_by_id(state, agreement_resolutions[0].chosen_meaning_id)

        assert getattr(meaning, "intro_kind", None) == "parenthetical_alias"
        assert getattr(meaning, "alias_target_text", None) == "This Master Services Agreement"

        alias_target_span = getattr(meaning, "alias_target_span", None)
        assert alias_target_span is not None
        assert alias_target_span[0] == "This Master Services Agreement"
        assert text[alias_target_span[1]:alias_target_span[2]] == "This Master Services Agreement"

    def test_parenthetical_alias_does_not_capture_trailing_prose(self):
        text = """
        This Master Services Agreement (the "Agreement") is entered into on the Effective Date.
        """.strip()

        det_cfg = DefinedTermDetectorConfig(
            unquoted_capitalised_terms_policy="always",
        )

        det_res, extr, reports, state = detect_and_resolve_terms(
            text,
            det_cfg=det_cfg,
            return_reports=True,
            return_state=True,
        )

        intro_keys = {i.normalized_key for i in det_res.introductions}
        assert "agreement" in intro_keys

        agreement_meaning = next(
            m
            for m in state.tier_1.meaning_index.values()
            if getattr(m, "normalized_key", None) == "agreement"
        )

        assert getattr(agreement_meaning, "intro_kind", None) == "parenthetical_alias"
        assert getattr(agreement_meaning, "alias_target_text", None) == "This Master Services Agreement"
        assert "is entered into on the Effective Date" not in agreement_meaning.alias_target_text

    def test_parenthetical_alias_preserves_traceability(self):
        text = """
        Acme Limited (the "Supplier") shall provide the Services.

        The Supplier shall perform the Services with reasonable skill and care.
        """.strip()

        det_cfg = DefinedTermDetectorConfig(
            unquoted_capitalised_terms_policy="always",
        )

        det_res, extr, reports, state = detect_and_resolve_terms(
            text,
            det_cfg=det_cfg,
            return_reports=True,
            return_state=True,
        )

        intro_keys = {i.normalized_key for i in det_res.introductions}
        assert "supplier" in intro_keys

        supplier_resolutions = resolutions_for_key(extr, "supplier")
        assert len(supplier_resolutions) >= 1
        assert supplier_resolutions[0].chosen_meaning_id is not None

        meaning = meaning_by_id(state, supplier_resolutions[0].chosen_meaning_id)

        assert getattr(meaning, "intro_kind", None) == "parenthetical_alias"
        assert getattr(meaning, "alias_target_text", None) == "Acme Limited"

        alias_target_span = getattr(meaning, "alias_target_span", None)
        assert alias_target_span is not None
        assert alias_target_span[0] == "Acme Limited"
        assert text[alias_target_span[1]:alias_target_span[2]] == "Acme Limited"

    def test_parenthetical_alias_extracts_schedule_style_antecedent(self):
        text = """
        Schedule A (the "Service Levels Schedule") forms part of this Agreement.

        The Service Levels Schedule shall apply to the support services.
        """.strip()

        det_cfg = DefinedTermDetectorConfig(
            unquoted_capitalised_terms_policy="always",
        )

        det_res, extr, reports, state = detect_and_resolve_terms(
            text,
            det_cfg=det_cfg,
            return_reports=True,
            return_state=True,
        )

        intro_keys = {i.normalized_key for i in det_res.introductions}
        assert "service_levels_schedule" in intro_keys

        schedule_resolutions = resolutions_for_key(extr, "service_levels_schedule")
        assert len(schedule_resolutions) >= 1
        assert schedule_resolutions[0].chosen_meaning_id is not None

        meaning = meaning_by_id(state, schedule_resolutions[0].chosen_meaning_id)

        assert getattr(meaning, "intro_kind", None) == "parenthetical_alias"
        assert getattr(meaning, "alias_target_text", None) == "Schedule A"

        alias_target_span = getattr(meaning, "alias_target_span", None)
        assert alias_target_span is not None
        assert alias_target_span[0] == "Schedule A"
        assert text[alias_target_span[1]:alias_target_span[2]] == "Schedule A"

    def test_means_definition_unchanged_by_alias_work(self):
        text = """
        "Services" means the implementation and support services.
        Acme Limited (the "Supplier") shall provide the Services.

        The Supplier shall provide the Services in accordance with this Agreement.
        """.strip()

        det_cfg = DefinedTermDetectorConfig(
            unquoted_capitalised_terms_policy="always",
        )

        det_res, extr, reports, state = detect_and_resolve_terms(
            text,
            det_cfg=det_cfg,
            return_reports=True,
            return_state=True,
        )

        service_resolutions = resolutions_for_key(extr, "services")
        supplier_resolutions = resolutions_for_key(extr, "supplier")

        assert len(service_resolutions) >= 1
        assert len(supplier_resolutions) >= 1
        assert service_resolutions[0].chosen_meaning_id is not None
        assert supplier_resolutions[0].chosen_meaning_id is not None

        service_meaning = meaning_by_id(state, service_resolutions[0].chosen_meaning_id)
        supplier_meaning = meaning_by_id(state, supplier_resolutions[0].chosen_meaning_id)

        assert "implementation and support services" in (getattr(service_meaning, "definition_text", "") or "").lower()

        assert getattr(supplier_meaning, "intro_kind", None) == "parenthetical_alias"
        assert getattr(supplier_meaning, "alias_target_text", None) == "Acme Limited"
