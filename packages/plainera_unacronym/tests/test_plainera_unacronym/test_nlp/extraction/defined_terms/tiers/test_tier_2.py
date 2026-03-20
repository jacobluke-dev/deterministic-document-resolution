from __future__ import annotations

from dataclasses import replace as dc_replace

import numpy as np
from plainera_unacronym.nlp.common.types import DefinedTermDetectorConfig
from plainera_unacronym.nlp.extraction.defined_terms.config import DefinedTermExtractionConfig
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


def _resolutions_for_key(extr, key: str):
    return [r for r in extr.term_resolutions if _resolution_key(r) == key]

class TestTier2:
    def test_tier2_applies_and_blended_scores_choose_winner(self, _patch):
        from plainera_unacronym.nlp.extraction.defined_terms.tiers import tier_2 as tier2_mod

        text = """
        "Services" means the consultancy services described in the main body.

        Schedule A
        "Services" means the software maintenance services described in this Schedule.

        In Schedule A, the Services include patching, bug fixes, and support updates.
        """.strip()

        det_cfg = DefinedTermDetectorConfig(
            unquoted_capitalised_terms_policy="legal_only"
        )

        def _fake_embed_texts(texts, model_name=None):
            rows = []
            for t in texts:
                t = t.lower()
                if "software maintenance services" in t:
                    rows.append([2.0, 0.0])  # candidate: term|services|2
                elif "consultancy services" in t:
                    rows.append([1.0, 0.0])  # candidate: term|services|1
                else:
                    rows.append([0.0, 0.0])  # context rows
            return np.array(rows, dtype=float)

        def _fake_cosine_sim01(ctx_vec, cand_mat_rows):
            sims = []
            for row in cand_mat_rows:
                marker = row[0]
                if marker == 2.0:
                    sims.append(0.95)  # prefer software maintenance
                elif marker == 1.0:
                    sims.append(0.10)  # de-prefer consultancy
                else:
                    sims.append(0.0)
            return np.array(sims, dtype=float)

        _patch(tier2_mod.rerank_term_occurrences_tier2, embed_texts=_fake_embed_texts)
        _patch(tier2_mod.rerank_term_occurrences_tier2, cosine_sim01=_fake_cosine_sim01)

        base_ext_cfg = DefinedTermExtractionConfig()
        ext_cfg = dc_replace(
            base_ext_cfg,
            tier2=dc_replace(
                base_ext_cfg.tier2,
                weight=1.0,
            ),
        )

        det_res, extr, reports, state = detect_and_resolve_terms(
            text,
            det_cfg=det_cfg,
            ext_cfg=ext_cfg,
            return_reports=True,
            return_state=True,
        )

        assert len(det_res.introductions) == 2
        assert len(det_res.mentions) == 1
        assert extr.ambiguous_keys == ("services",)

        assert state.tier_2.report is not None
        assert state.tier_2.report.applied == 1
        assert state.tier_2.report.skipped == 0

        assert len(state.tier_2.ranked) == 1
        assert state.tier_2.ranked[0].applied is True
        assert state.tier_2.ranked[0].blended_scores is not None
        assert state.tier_2.ranked[0].tier2_sims is not None

        service_resolutions = _resolutions_for_key(extr, "services")
        assert len(service_resolutions) == 1

        resolution = service_resolutions[0]
        assert resolution.chosen_meaning_id == "term|services|2"
        assert resolution.resolution_method == "tier2_blend"

        scores_by_id = {c.meaning_id: c for c in resolution.candidate_scores}
        assert scores_by_id["term|services|2"].tier2_score > scores_by_id["term|services|1"].tier2_score
        assert scores_by_id["term|services|2"].total_score > scores_by_id["term|services|1"].total_score

    def test_e2e_tier2_applies_and_changes_final_resolution(self, _patch):
        from plainera_unacronym.nlp.extraction.defined_terms.tiers import tier_2 as tier2_mod

        text = """
        "Services" means the consultancy services described in the main body.

        Schedule A
        "Services" means the software maintenance services described in this Schedule.

        In Schedule A, the Services include patching, bug fixes, and support updates.
        """.strip()

        det_cfg = DefinedTermDetectorConfig(
            unquoted_capitalised_terms_policy="legal_only"
        )

        def _fake_embed_texts(texts, model_name=None):
            rows = []
            for t in texts:
                t = t.lower()
                if "software maintenance services" in t:
                    rows.append([2.0, 0.0])  # candidate: term|services|2
                elif "consultancy services" in t:
                    rows.append([1.0, 0.0])  # candidate: term|services|1
                else:
                    rows.append([0.0, 0.0])  # context rows
            return np.array(rows, dtype=float)

        def _fake_cosine_sim01(ctx_vec, cand_mat_rows):
            sims = []
            for row in cand_mat_rows:
                marker = row[0]
                if marker == 2.0:
                    sims.append(0.95)  # prefer software maintenance
                elif marker == 1.0:
                    sims.append(0.10)  # de-prefer consultancy
                else:
                    sims.append(0.0)
            return np.array(sims, dtype=float)

        _patch(tier2_mod.rerank_term_occurrences_tier2, embed_texts=_fake_embed_texts)
        _patch(tier2_mod.rerank_term_occurrences_tier2, cosine_sim01=_fake_cosine_sim01)

        base_ext_cfg = DefinedTermExtractionConfig()
        ext_cfg = dc_replace(
            base_ext_cfg,
            tier2=dc_replace(
                base_ext_cfg.tier2,
                weight=1.0,
            ),
        )

        det_res, extr, reports, state = detect_and_resolve_terms(
            text,
            det_cfg=det_cfg,
            ext_cfg=ext_cfg,
            return_reports=True,
            return_state=True,
        )

        assert len(det_res.introductions) == 2
        assert len(det_res.mentions) == 1
        assert extr.ambiguous_keys == ("services",)

        # Tier-2 genuinely ran for this occurrence.
        assert state.tier_2.report is not None
        assert state.tier_2.report.applied == 1
        assert state.tier_2.report.skipped == 0
        assert len(state.tier_2.ranked) == 1
        assert state.tier_2.ranked[0].applied is True
        assert state.tier_2.ranked[0].skip_reason is None
        assert state.tier_2.ranked[0].tier2_sims is not None
        assert state.tier_2.ranked[0].blended_scores is not None

        service_resolutions = _resolutions_for_key(extr, "services")
        assert len(service_resolutions) == 1

        resolution = service_resolutions[0]
        assert resolution.resolution_method == "tier2_blend"
        assert resolution.chosen_meaning_id == "term|services|2"

        scores_by_id = {c.meaning_id: c for c in resolution.candidate_scores}
        assert scores_by_id["term|services|2"].tier2_score is not None
        assert scores_by_id["term|services|1"].tier2_score is not None
        assert scores_by_id["term|services|2"].tier2_score > scores_by_id["term|services|1"].tier2_score
        assert scores_by_id["term|services|2"].total_score > scores_by_id["term|services|1"].total_score

        # Optional: nice report-level sanity check.
        tier2_stage = next(r for r in reports if r.name == "tier2_term_semantic_rerank")
        assert "applied=1" in tier2_stage.info
