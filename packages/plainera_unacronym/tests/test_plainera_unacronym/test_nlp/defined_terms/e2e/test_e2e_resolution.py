from __future__ import annotations

import types

from plainera_unacronym.nlp.common.types import DefinedTermDetectorConfig
from plainera_unacronym.nlp.extraction.defined_terms.execute import detect_and_resolve_terms


def _resolution_key(r) -> str | None:
    if hasattr(r, "normalized_key"):
        return r.normalized_key
    if hasattr(r, "term_key"):
        return r.term_key
    if hasattr(r, "key"):
        return r.key

    occ = getattr(r, "occurrence", None)
    if occ is not None and hasattr(occ, "normalized_key"):
        return occ.normalized_key

    return None


def _chosen_sense_ids_for_key(extr, key: str) -> list[str]:
    return [
        r.chosen_sense_id
        for r in _resolutions_for_key(extr, key)
        if getattr(r, "chosen_sense_id", None) is not None
    ]


def _resolutions_for_key(extr, key: str):
    return [r for r in extr.term_resolutions if _resolution_key(r) == key]


def _sense_text_by_id(state) -> dict[str, str]:
    out: dict[str, str] = {}

    for sense_id, sense in state.tier_1.sense_index.items():
        definition_text = getattr(sense, "definition_text", None)

        if not definition_text:
            for entry in state.definition_entries:
                if getattr(entry, "sense_id", None) == sense_id:
                    definition_text = getattr(entry, "definition_text", None)
                    break

        out[sense_id] = (definition_text or "").lower()

    return out


class TestDefinedTermResolutionE2E:
    def test_services_disambiguation_by_section(self):
        text = """
        "Services" means the consultancy services described in the main body.

        The Services shall be delivered by the Supplier in accordance with the main body of this Agreement.

        Schedule A
        "Services" means the software maintenance services described in this Schedule.

        In Schedule A, the Services include patching, bug fixes, and support updates.
        """.strip()

        det_cfg = DefinedTermDetectorConfig(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False,
        )

        det_res, extr, reports, state = detect_and_resolve_terms(
            text,
            det_cfg=det_cfg,
            return_reports=True,
            return_state=True,
        )

        assert len(det_res.introductions) == 2

        chosen_ids = _chosen_sense_ids_for_key(extr, "services")
        assert len(chosen_ids) >= 2

        sense_text = _sense_text_by_id(state)
        chosen_texts = [sense_text[sid] for sid in chosen_ids]

        assert any("consultancy services" in txt for txt in chosen_texts)
        assert any("software maintenance services" in txt for txt in chosen_texts)

    def test_single_candidate_term_resolves_deterministically(self):
        text = """
        "Effective Date" means the date on which both Parties sign this Agreement.

        The Effective Date shall be recorded in writing.
        """.strip()

        det_cfg = DefinedTermDetectorConfig(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False,
        )

        det_res, extr, reports, state = detect_and_resolve_terms(
            text,
            det_cfg=det_cfg,
            return_reports=True,
            return_state=True,
        )

        assert len(det_res.introductions) == 1

        effective_resolutions = _resolutions_for_key(extr, "effective_date")
        assert len(effective_resolutions) == 1
        assert effective_resolutions[0].chosen_sense_id is not None

        assert extr.undecided == []
        assert "effective_date" not in extr.ambiguous_keys

        sense_id = effective_resolutions[0].chosen_sense_id
        sense_text = _sense_text_by_id(state)

        assert "both parties sign this agreement" in sense_text[sense_id]

    def test_tier2_skip_when_confident(self):
        text = """
        "Effective Date" means the date on which both Parties sign this Agreement.

        The Effective Date shall be recorded in writing.
        """.strip()

        det_cfg = DefinedTermDetectorConfig(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False,
        )

        det_res, extr, reports, state = detect_and_resolve_terms(
            text,
            det_cfg=det_cfg,
            return_reports=True,
            return_state=True,
        )
        assert len(det_res.introductions) == 1
        assert len(_resolutions_for_key(extr, "effective_date")) == 1

        assert state.tier_2.report is not None
        assert state.tier_2.report.applied == 0
        assert state.tier_2.report.skipped == 1
        assert state.tier_2.report.reasons == {"single_candidate": 1}

    def test_model_unavailable_fallback(self, _patch):
        from plainera_unacronym.nlp.extraction.defined_terms import stage_funcs

        def _fake_tier2(*, text, t1_ranked, sense_index, cfg):
            report = types.SimpleNamespace(
                applied=False,
                reason="model_unavailable",
            )
            return (), report

        _patch(stage_funcs.st_tier2_term_semantic_rerank, rerank_term_occurrences_tier2=_fake_tier2)

        text = """
        "Effective Date" means the date on which both Parties sign this Agreement.

        The Effective Date shall be recorded in writing.
        """.strip()
        det_cfg = DefinedTermDetectorConfig(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False,
        )

        det_res, extr, reports, state = detect_and_resolve_terms(
            text,
            det_cfg=det_cfg,
            return_reports=True,
            return_state=True,
        )
        assert len(det_res.introductions) == 1

        effective_resolutions = _resolutions_for_key(extr, "effective_date")
        assert len(effective_resolutions) == 1
        assert effective_resolutions[0].chosen_sense_id is not None

        assert state.tier_2.report is not None
        assert state.tier_2.report.applied is False
        assert getattr(state.tier_2.report, "reason", None) == "model_unavailable"

        assert extr.undecided == []
        assert "effective_date" not in extr.ambiguous_keys

    def test_no_later_mentions_only_introductions(self):
        text = """
        "Effective Date" means the date on which both Parties sign this Agreement.
        """.strip()

        det_res, extr, reports, state = detect_and_resolve_terms(
            text,
            return_reports=True,
            return_state=True,
        )

        assert len(det_res.introductions) == 1
        assert det_res.mentions == []

        assert len(state.definition_entries) == 1
        assert len(state.tier_1.sense_index) == 1
        assert len(state.tier_1.occurrences) == 0
        assert len(state.tier_1.ranked) == 0

        assert extr.term_resolutions == []
        assert extr.undecided == []
        assert extr.ambiguous_keys == ()

    def test_ambiguous_term_with_no_strong_winner_stays_unresolved(self, _patch):
        from plainera_unacronym.nlp.extraction.defined_terms import stage_funcs

        def _fake_tier2(*, text, t1_ranked, sense_index, cfg):
            ranked = [
                types.SimpleNamespace(
                    occ=r.occ,
                    applied=False,
                    skip_reason="test_no_semantic_rescue",
                    tier2_sims=None,
                    blended_scores=None,
                )
                for r in t1_ranked
            ]
            report = types.SimpleNamespace(
                applied=0,
                skipped=len(t1_ranked),
                reasons={"test_no_semantic_rescue": len(t1_ranked)},
            )
            return tuple(ranked), report

        _patch(
            stage_funcs.st_tier2_term_semantic_rerank,
            rerank_term_occurrences_tier2=_fake_tier2,
        )

        text = """
        "Services" means the services described in Part 1.

        Part 2
        "Services" means the services described in Part 2.

        The Services shall be provided as agreed.
        """.strip()

        det_cfg = DefinedTermDetectorConfig(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False,
        )

        det_res, extr, reports, state = detect_and_resolve_terms(
            text,
            det_cfg=det_cfg,
            return_reports=True,
            return_state=True,
        )

        assert len(det_res.introductions) == 2
        assert len(det_res.mentions) == 1
        assert extr.ambiguous_keys == ("services",)

        service_resolutions = _resolutions_for_key(extr, "services")
        assert len(service_resolutions) == 1
        assert service_resolutions[0].chosen_sense_id is None
        assert service_resolutions[0] in extr.undecided

    # TODO AS PART OF TICKET 96
    # def test_prefer_prior_definition_when_context_is_otherwise_equal(self, _patch):
    #     from plainera_unacronym.nlp.extraction.defined_terms import stage_funcs
    #
    #     original = stage_funcs.score_term_occurrences_tier1
    #
    #     def _fake_tier1(*, text, occurrences, term_sense_index, structure_index, cfg):
    #         ranked = original(
    #             text=text,
    #             occurrences=occurrences,
    #             term_sense_index=term_sense_index,
    #             structure_index=structure_index,
    #             cfg=cfg,
    #         )
    #
    #         adjusted = []
    #         for r in ranked:
    #             if r.occ.normalized_key != "services":
    #                 adjusted.append(r)
    #                 continue
    #
    #             adjusted_scores = {sense_id: 1.0 for sense_id in r.candidate_scores}
    #             adjusted.append(
    #                 types.SimpleNamespace(
    #                     occ=r.occ,
    #                     candidate_scores=adjusted_scores,
    #                     chosen_sense_id=None,
    #                 )
    #             )
    #         return tuple(adjusted)
    #
    #     def _fake_tier2(*, text, t1_ranked, sense_index, cfg):
    #         ranked = [
    #             types.SimpleNamespace(
    #                 occ=r.occ,
    #                 applied=False,
    #                 skip_reason="test_keep_tier1",
    #                 tier2_sims=None,
    #                 blended_scores=None,
    #             )
    #             for r in t1_ranked
    #         ]
    #         report = types.SimpleNamespace(
    #             applied=0,
    #             skipped=len(t1_ranked),
    #             reasons={"test_keep_tier1": len(t1_ranked)},
    #         )
    #         return tuple(ranked), report
    #
    #     _patch(
    #         stage_funcs.st_tier1_score_term_occurrences,
    #         score_term_occurrences_tier1=_fake_tier1,
    #     )
    #     _patch(
    #         stage_funcs.st_tier2_term_semantic_rerank,
    #         rerank_term_occurrences_tier2=_fake_tier2,
    #     )
    #
    #     text = """
    #     "Services" means the services described in the first section.
    #
    #     "Services" means the services described in the second section.
    #
    #     The Services shall be delivered promptly.
    #     """.strip()
    #
    #     det_cfg = DefinedTermDetectorConfig(
    #         allow_unquoted_capitalised_terms=True,
    #         require_legal_domain_for_unquoted=False,
    #     )
    #
    #     det_res, extr, reports, state = detect_and_resolve_terms(
    #         text,
    #         det_cfg=det_cfg,
    #         return_reports=True,
    #         return_state=True,
    #     )
    #
    #     chosen_ids = _chosen_sense_ids_for_key(extr, "services")
    #     assert len(chosen_ids) == 1
    #     assert chosen_ids[0] == "term|services|1"

    def test_structure_proximity_beats_lexical_similarity(self):
        text = """
        "Services" means the consultancy services described in the main body.

        The parties agree that the consultancy work is documented in the main body.

        Schedule A
        "Services" means the maintenance services described in this Schedule.

        In Schedule A, the Services shall include patching and updates.
        """.strip()

        det_cfg = DefinedTermDetectorConfig(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False,
        )

        det_res, extr, reports, state = detect_and_resolve_terms(
            text,
            det_cfg=det_cfg,
            return_reports=True,
            return_state=True,
        )

        assert len(det_res.introductions) == 2
        assert len(det_res.mentions) == 1
        assert extr.ambiguous_keys == ("services",)

        chosen_ids = _chosen_sense_ids_for_key(extr, "services")
        assert len(chosen_ids) == 1

        sense_text = _sense_text_by_id(state)
        assert "maintenance services" in sense_text[chosen_ids[0]]

    def test_quoted_later_mention_resolves(self):
        text = """
        "Confidential Information" means any non-public technical, commercial, or business information.

        Each Party shall protect "Confidential Information" from unauthorised disclosure.
        """.strip()

        det_res, extr, reports, state = detect_and_resolve_terms(
            text,
            return_reports=True,
            return_state=True,
        )

        assert len(det_res.introductions) == 1
        assert len(det_res.mentions) == 1

        conf_resolutions = _resolutions_for_key(extr, "confidential_information")
        assert len(conf_resolutions) == 1
        assert conf_resolutions[0].chosen_sense_id == "term|confidential_information|1"
        assert extr.undecided == []

    def test_parenthetical_alias_intro_participates_in_resolution(self):
        text = """
        This Master Services Agreement (the "Agreement") is entered into on the Effective Date.
        "Effective Date" means the date on which both Parties sign this Agreement.

        The Agreement shall commence on the Effective Date.
        """.strip()

        det_cfg = DefinedTermDetectorConfig(
            allow_unquoted_capitalised_terms=True,
            require_legal_domain_for_unquoted=False,
        )

        det_res, extr, reports, state = detect_and_resolve_terms(
            text,
            det_cfg=det_cfg,
            return_reports=True,
            return_state=True,
        )

        intro_keys = {i.normalized_key for i in det_res.introductions}
        assert "agreement" in intro_keys

        agreement_resolutions = _resolutions_for_key(extr, "agreement")
        assert len(agreement_resolutions) == 2
        assert agreement_resolutions[0].chosen_sense_id == "term|agreement|1"
        assert agreement_resolutions[0].resolution_method in {"tier1", "tier2_blend"}
