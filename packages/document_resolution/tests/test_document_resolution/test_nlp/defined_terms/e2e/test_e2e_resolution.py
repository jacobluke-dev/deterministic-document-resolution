from __future__ import annotations

import types

from document_resolution.nlp.common.types import DefinedTermDetectorConfig
from document_resolution.nlp.detection.defined_terms import DefinedTermDetector, DefinedTermMention
from document_resolution.nlp.extraction.base.base_execute import run_flow_with_options
from document_resolution.nlp.extraction.defined_terms.execute import detect_and_resolve_terms
from document_resolution.nlp.extraction.defined_terms.extract_flow import DefinedTermResolutionFlow
from document_resolution.nlp.extraction.defined_terms.state import TermFlowState
from document_resolution.nlp.extraction.defined_terms.types import (
    TermCandidateScore,
    TermResolution,
    TermTier1OccurrenceRanking,
)
from test_document_resolution.test_nlp.defined_terms.e2e.defined_terms_e2e_common import (
    chosen_meaning_ids_for_key,
    meaning_text_by_id,
    resolutions_for_key,
)


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
            unquoted_capitalised_terms_policy="always"
        )

        det_res, extr, reports, state = detect_and_resolve_terms(
            text,
            det_cfg=det_cfg,
            return_reports=True,
            return_state=True,
        )

        assert len(det_res.introductions) == 2

        chosen_ids = chosen_meaning_ids_for_key(extr, "services")
        assert len(chosen_ids) >= 2

        meaning_text = meaning_text_by_id(state)
        chosen_texts = [meaning_text[sid] for sid in chosen_ids]

        assert any("consultancy services" in txt for txt in chosen_texts)
        assert any("software maintenance services" in txt for txt in chosen_texts)

    def test_single_candidate_term_resolves_deterministically(self):
        text = """
        "Effective Date" means the date on which both Parties sign this Agreement.

        The Effective Date shall be recorded in writing.
        """.strip()

        det_cfg = DefinedTermDetectorConfig(
            unquoted_capitalised_terms_policy="always"
        )

        det_res, extr, reports, state = detect_and_resolve_terms(
            text,
            det_cfg=det_cfg,
            return_reports=True,
            return_state=True,
        )

        assert len(det_res.introductions) == 1

        effective_resolutions = resolutions_for_key(extr, "effective_date")
        assert len(effective_resolutions) == 2
        assert effective_resolutions[0].chosen_meaning_id is not None

        assert extr.undecided == []
        assert "effective_date" not in extr.ambiguous_keys

        meaning_id = effective_resolutions[0].chosen_meaning_id
        meaning_text = meaning_text_by_id(state)

        assert "both parties sign this agreement" in meaning_text[meaning_id]

    def test_tier2_skip_when_confident(self):
        text = """
        "Effective Date" means the date on which both Parties sign this Agreement.

        The Effective Date shall be recorded in writing.
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
        assert len(det_res.introductions) == 1
        assert len(resolutions_for_key(extr, "effective_date")) == 2

        assert state.tier_2.report is not None
        assert state.tier_2.report.applied == 0
        assert state.tier_2.report.skipped == 2
        assert state.tier_2.report.reasons == {"single_candidate": 2}

    def test_model_unavailable_fallback(self, _patch):
        from document_resolution.nlp.extraction.defined_terms import stage_funcs

        def _fake_tier2(*, text, t1_ranked, meaning_index, cfg):
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
            unquoted_capitalised_terms_policy="always",
        )

        det_res, extr, reports, state = detect_and_resolve_terms(
            text,
            det_cfg=det_cfg,
            return_reports=True,
            return_state=True,
        )
        assert len(det_res.introductions) == 1

        effective_resolutions = resolutions_for_key(extr, "effective_date")
        assert len(effective_resolutions) == 2
        assert effective_resolutions[0].chosen_meaning_id is not None

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
        assert det_res.mentions == [DefinedTermMention(term='Effective Date',
                    start_offset=1,
                    end_offset=15,
                    normalized_key='effective_date',
                    confidence=1.0,
                    segment_window=None)]

        assert len(state.definition_entries) == 1
        assert len(state.tier_1.meaning_index) == 1
        assert len(state.tier_1.occurrences) == 1
        assert len(state.tier_1.ranked) == 1

        assert extr.term_resolutions == [TermResolution(occurrence_span=('Effective Date', 1, 15),
                term='Effective Date',
                normalized_key='effective_date',
                chosen_meaning_id='term|effective_date|1',
                chosen_definition_span=('the date on which both Parties sign '
                                        'this Agreement',
                                        23,
                                        73),
                candidate_scores=(TermCandidateScore(meaning_id='term|effective_date|1',
                                                     total_score=5.0,
                                                     tier1_score=5.0,
                                                     tier2_score=None,
                                                     definition_span=('the '
                                                                      'date on '
                                                                      'which '
                                                                      'both '
                                                                      'Parties '
                                                                      'sign '
                                                                      'this '
                                                                      'Agreement',
                                                                      23,
                                                                      73),
                                                     components={}),),
                resolution_method='tier1')]
        assert extr.undecided == []
        assert extr.ambiguous_keys == ()

    def test_ambiguous_term_with_no_strong_winner_stays_unresolved(self, _patch):
        from document_resolution.nlp.extraction.defined_terms import stage_funcs

        def _fake_tier2(*, text, t1_ranked, meaning_index, cfg):
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
            unquoted_capitalised_terms_policy="always",
        )

        det_res, extr, reports, state = detect_and_resolve_terms(
            text,
            det_cfg=det_cfg,
            return_reports=True,
            return_state=True,
        )

        assert len(det_res.introductions) == 2
        assert len(det_res.mentions) == 3
        assert extr.ambiguous_keys == ("services",)

        service_resolutions = resolutions_for_key(extr, "services")
        assert len(service_resolutions) == 3

        assert service_resolutions[0].chosen_meaning_id == "term|services|1"
        assert service_resolutions[0].resolution_method == "tier1"

        assert service_resolutions[1].chosen_meaning_id is None
        assert service_resolutions[1].resolution_method == "unresolved"

        assert service_resolutions[2].chosen_meaning_id is None
        assert service_resolutions[2].resolution_method == "unresolved"

    def test_prefer_prior_definition_when_context_is_otherwise_equal(self, _patch):
        from document_resolution.nlp.extraction.defined_terms import stage_funcs

        original = stage_funcs.score_term_occurrences_tier1

        def _fake_tier1(*, text, occurrences, term_meaning_index, structure_index, cfg):
            ranked = original(
                text=text,
                occurrences=occurrences,
                term_meaning_index=term_meaning_index,
                structure_index=structure_index,
                cfg=cfg,
            )

            adjusted = []
            for r in ranked:
                if r.occ.normalized_key != "services":
                    adjusted.append(r)
                    continue

                adjusted_scores = {meaning_id: 1.0 for meaning_id in r.candidate_scores}
                adjusted.append(
                    TermTier1OccurrenceRanking(
                        occ=r.occ,
                        candidate_scores=adjusted_scores,
                        chosen_meaning_id="term|services|1",
                        gap=0.0,
                        margin=0.0,
                    )
                )
            return tuple(adjusted)

        def _fake_tier2(*, text, t1_ranked, meaning_index, cfg):
            ranked = [
                types.SimpleNamespace(
                    occ=r.occ,
                    chosen_meaning_id=r.chosen_meaning_id,
                    applied=False,
                    skip_reason="test_keep_tier1",
                    tier2_sims=None,
                    blended_scores=None,
                )
                for r in t1_ranked
            ]
            report = types.SimpleNamespace(
                applied=0,
                skipped=len(t1_ranked),
                reasons={"test_keep_tier1": len(t1_ranked)},
            )
            return tuple(ranked), report

        _patch(
            stage_funcs.st_tier1_score_term_occurrences,
            score_term_occurrences_tier1=_fake_tier1,
        )
        _patch(
            stage_funcs.st_tier2_term_semantic_rerank,
            rerank_term_occurrences_tier2=_fake_tier2,
        )

        text = """
        "Services" means the services described in the first section.

        "Services" means the services described in the second section.

        The Services shall be delivered promptly.
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

        tier1_rankings = [r for r in state.tier_1.ranked if r.occ.normalized_key == "services"]
        assert len(tier1_rankings) == 3

        later_ranking = max(tier1_rankings, key=lambda r: r.occ.start_offset)
        assert later_ranking.chosen_meaning_id == "term|services|1"

    def test_structure_proximity_beats_lexical_similarity(self):
        text = """
        "Services" means the consultancy services described in the main body.

        The parties agree that the consultancy work is documented in the main body.

        Schedule A
        "Services" means the maintenance services described in this Schedule.

        In Schedule A, the Services shall include patching and updates.
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

        assert len(det_res.introductions) == 2
        assert len(det_res.mentions) == 3
        assert extr.ambiguous_keys == ("services",)

        services_resolutions = resolutions_for_key(extr, "services")
        assert len(services_resolutions) == 3

        meaning_text = meaning_text_by_id(state)

        schedule_a_resolution = services_resolutions[-1]
        assert schedule_a_resolution.resolution_method == "tier1"
        assert schedule_a_resolution.chosen_meaning_id is not None
        assert "maintenance services" in meaning_text[schedule_a_resolution.chosen_meaning_id]

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
        assert len(det_res.mentions) == 2

        conf_resolutions = resolutions_for_key(extr, "confidential_information")
        assert len(conf_resolutions) == 2
        assert conf_resolutions[0].chosen_meaning_id == "term|confidential_information|1"
        assert extr.undecided == []

    def test_parenthetical_alias_intro_participates_in_resolution(self):
        text = """
        This Master Services Agreement (the "Agreement") is entered into on the Effective Date.
        "Effective Date" means the date on which both Parties sign this Agreement.

        The Agreement shall commence on the Effective Date.
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
        assert len(agreement_resolutions) == 3
        assert agreement_resolutions[0].chosen_meaning_id == "term|agreement|1"
        assert agreement_resolutions[0].resolution_method in {"tier1", "tier2_blend"}

    def test_detect_and_resolve_terms_returns_trace_events_when_trace_enabled(self):
        text = '"Agreement" means this agreement. The Agreement shall apply.'

        det_res, extr, reports, trace_events = detect_and_resolve_terms(
            text,
            return_reports=True,
            trace=True,
        )

        assert det_res is not None
        assert extr is not None
        assert reports
        assert trace_events is not None
        assert isinstance(trace_events, list)

    def test_detect_and_resolve_terms_returns_state_and_trace_events_when_requested(self):
        text = '"Agreement" means this agreement. The Agreement shall apply.'

        det_res, extr, reports, state, trace_events = detect_and_resolve_terms(
            text,
            return_reports=True,
            return_state=True,
            trace=True,
        )

        assert state.det_res is det_res
        assert state.extr is extr
        assert trace_events is not None
        assert isinstance(trace_events, list)

    def test_detect_and_resolve_terms_without_trace_does_not_return_trace_events(self):
        text = '"Agreement" means this agreement. The Agreement shall apply.'

        out = detect_and_resolve_terms(
            text,
            return_reports=True,
            trace=False,
        )

        assert len(out) == 3
        det_res, extr, reports = out
        assert det_res is not None
        assert extr is not None
        assert reports

    def test_detect_and_resolve_terms_trace_filter_restricts_trace_output(self):
        text = (
            '"Agreement" means this agreement. '
            '"Services" means the services. '
            'The Agreement and the Services shall apply.'
        )

        _, _, reports_all, trace_all = detect_and_resolve_terms(
            text,
            return_reports=True,
            trace=True,
        )
        _, _, reports_filtered, trace_filtered = detect_and_resolve_terms(
            text,
            return_reports=True,
            trace=True,
            trace_filter=r"^agreement$",
        )

        assert reports_all
        assert reports_filtered
        assert trace_all is not None
        assert trace_filtered is not None
        assert len(trace_filtered) <= len(trace_all)

    def test_run_flow_with_options_sets_flow_trace_events(self):
        flow = DefinedTermResolutionFlow(trace=True, trace_filter=r"^agreement$")
        state = TermFlowState(
            text='"Agreement" means this agreement. The Agreement shall apply.',
            det_cfg=flow.det_cfg,
            ext_cfg=flow.ext_cfg,
        )

        det_res, extr, reports, trace_events = run_flow_with_options(
            flow=flow,
            state=state,
            return_reports=True,
            trace=True,
        )

        assert flow.trace_events == trace_events

    def test_detect_and_resolve_terms_emits_later_defined_term_mentions(self):
        text = (
            'This Agreement is made between the Supplier and the Customer. '
            '"Services" means the consulting services described in Schedule 1. '
            'The Supplier shall provide the Services to the Customer.'
        )

        det, ext, _ = detect_and_resolve_terms(text, return_reports=True)

        assert det.introductions != []
        assert det.mentions != []
        assert ext.term_resolutions != []

def test_detector_emits_later_mention_for_defined_term():
    text = (
        'This Agreement is made between the Supplier and the Customer. '
        '"Services" means the consulting services described in Schedule 1. '
        'The Supplier shall provide the Services to the Customer.'
    )

    result = DefinedTermDetector(DefinedTermDetectorConfig()).detect(text)

    assert len(result.introductions) == 1
    assert len(result.mentions) == 2

    services_mentions = [m for m in result.mentions if m.normalized_key == "services"]
    assert len(services_mentions) == 2

    intro = result.introductions[0]
    later_mentions = [m for m in services_mentions if m.start_offset >= intro.end_offset]
    assert len(later_mentions) == 1


def test_defined_term_later_exact_reference_effective_date_is_returned():
    text = (
        '"Effective Date" means 1 April 2026. '
        'The Supplier shall commence delivery on the Effective Date.'
    )

    det_res, term_result = detect_and_resolve_terms(text)

    assert [intro.term for intro in det_res.introductions] == ["Effective Date"]
    assert "Effective Date" in [mention.term for mention in det_res.mentions]
    assert "effective_date" in [res.normalized_key for res in term_result.term_resolutions]


def test_defined_term_plural_later_reference_resolves_to_known_singular_term():
    text = (
        '"Business Day" means any day other than a Saturday or Sunday. '
        'The parties must respond within 5 Business Days.'
    )

    det_res, term_result = detect_and_resolve_terms(text)

    assert [intro.term for intro in det_res.introductions] == ["Business Day"]
    terms = [mention.term for mention in det_res.mentions]
    assert len(terms) == 2
    assert "Business Days" in [mention.term for mention in det_res.mentions]
    assert "business_day" in [res.normalized_key for res in term_result.term_resolutions]


def test_defined_term_with_no_later_occurrence_emits_intro_resolution():
    text = '"Data Protection Laws" means all applicable privacy legislation.'

    det_res, term_result = detect_and_resolve_terms(text)

    assert [intro.term for intro in det_res.introductions] == ["Data Protection Laws"]
    assert [res.term for res in term_result.term_resolutions] == ["Data Protection Laws"]


def test_defined_term_intro_emits_resolution_even_when_only_pre_definition_occurrence_exists():
    text = (
        "The Supplier shall provide all Deliverables. "
        '"Deliverables" means all reports and outputs produced under this Agreement.'
    )

    det_res, term_result = detect_and_resolve_terms(text)

    assert [intro.term for intro in det_res.introductions] == ["Deliverables"]
    assert [res.term for res in term_result.term_resolutions] == ["Deliverables"]
