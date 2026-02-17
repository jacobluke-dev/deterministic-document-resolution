from dataclasses import replace

import numpy as np

from plainera_unacronym.nlp.common.types import DetectorConfig
from plainera_unacronym.nlp.extraction.config import ExtractionConfig, Tier2Config  # adjust import if needed
from plainera_unacronym.nlp.extraction.engine.detect_flow import ExtractionFlow
from plainera_unacronym.nlp.extraction.engine.state import FlowState
from plainera_unacronym.nlp.extraction.engine import stage_funcs as f
from plainera_unacronym.nlp.extraction.tiers import tier_2 as Tier2
from plainera_unacronym.nlp.common.types import OccurrenceLite, AcronymSense
from plainera_unacronym.nlp.extraction.tiers.types import Tier1OccurrenceRanking


def _mk_state(*, enabled: bool) -> FlowState:
    ext_cfg = replace(ExtractionConfig(), tier2=Tier2Config(enabled=enabled, weight=0.5, model_name="fake"))
    s = FlowState(text="ctx ... GPU ... kernel ...", det_cfg=DetectorConfig(), ext_cfg=ext_cfg)
    s.det_res = object()  # only asserted as not None in these stages
    return s

class TestSrTier2SemanticRerank:
    def test_tier2_disabled_sets_report_and_no_rankings(self,):
        s = _mk_state(enabled=False)

        # Set minimal Tier-1 ranked list (so "skipped=n" is meaningful)
        t1 = s.disambig.tier1
        t1.ranked = [
            Tier1OccurrenceRanking(
                occ=OccurrenceLite("GPU", 5, 8),
                candidate_scores={"gpu|graphics": 0.5, "gpu|general": 0.49},
                chosen_sense_id=None,
                gap=0.01,
                margin=0.02,
            )
        ]

        f.st_tier2_semantic_rerank(s, window_chars=50, auto_margin_ceiling=0.02)

        assert s.disambig.tier2.report is not None
        assert s.disambig.tier2.report.applied == 0
        assert s.disambig.tier2.report.reasons["disabled"] == 1
        assert s.disambig.tier2.ranked == []


    def test_tier2_model_unavailable_falls_back_cleanly(self, _patch):
        s = _mk_state(enabled=True)

        # Seed Tier-1 work
        t1 = s.disambig.tier1
        t1.sense_index = {
            "gpu|graphics": AcronymSense("GPU", "Graphics Processing Unit", "gpu|graphics", 0.8, [], 1),
            "gpu|general": AcronymSense("GPU", "General Purpose Unit", "gpu|general", 0.7, [], 1),
        }
        t1.ranked = [
            Tier1OccurrenceRanking(
                occ=OccurrenceLite("GPU", 5, 8),
                candidate_scores={"gpu|graphics": 0.5, "gpu|general": 0.5},
                chosen_sense_id=None,
                gap=0.0,
                margin=0.0,
            )
        ]

        # Force embedder failure
        _patch(f.st_tier2_semantic_rerank,embed_texts=lambda *a, **k: None)

        f.st_tier2_semantic_rerank(s, window_chars=50)

        rep = s.disambig.tier2.report
        assert rep is not None
        assert rep.applied == 0
        assert rep.reasons["model_unavailable"] == 1
        assert s.disambig.tier2.ranked[0].applied is False
        assert s.disambig.tier2.ranked[0].skip_reason == "model_unavailable"


    def test_tier2_applies_and_blends_in_tier1_order(self, monkeypatch):
        s = _mk_state(enabled=True)
        s.text = "kernel launch overhead ... GPU ..."

        t1 = s.disambig.tier1
        t1.sense_index = {
            "gpu|graphics": AcronymSense("GPU", "Graphics Processing Unit", "gpu|graphics", 0.8, [], 1),
            "gpu|general": AcronymSense("GPU", "General Purpose Unit", "gpu|general", 0.7, [], 1),
        }
        t1.ranked = [
            Tier1OccurrenceRanking(
                occ=OccurrenceLite("GPU", 10, 13),
                candidate_scores={"gpu|graphics": 0.40, "gpu|general": 0.60},
                chosen_sense_id=None,
                gap=0.20,
                margin=0.0,
            )
        ]

        def fake_embed_texts(_model_name, texts):
            # Called twice: candidates (sorted unique), then contexts
            vecs = []
            for t in texts:
                if "Graphics Processing Unit" in t:
                    vecs.append([1.0, 0.0])
                elif "General Purpose Unit" in t:
                    vecs.append([0.0, 1.0])
                else:
                    # context -> align with graphics
                    vecs.append([1.0, 0.0])
            return np.asarray(vecs, dtype=float)

        # Patch EXACTLY where the stage looks it up
        monkeypatch.setattr(Tier2, "embed_texts", fake_embed_texts, raising=True)

        f.st_tier2_semantic_rerank(s, window_chars=50)

        r2 = s.disambig.tier2.ranked[0]
        assert r2.applied is True
        assert r2.blended_scores is not None
        assert list(r2.blended_scores.keys()) == ["gpu|graphics", "gpu|general"]
        assert r2.blended_scores["gpu|graphics"] > r2.blended_scores["gpu|general"]


class TestStTier1SelectAndAssemble:
    def test_select_and_assemble_uses_tier1_when_tier2_absent(self):
        ext_cfg = replace(ExtractionConfig(), tier2=Tier2Config(enabled=False))
        s = FlowState(text="x", det_cfg=DetectorConfig(), ext_cfg=ext_cfg)
        s.det_res = object()

        t1 = s.disambig.tier1
        t1.senses_by_acronym = {
            "GPU": [
                AcronymSense("GPU", "Graphics Processing Unit", "gpu|graphics", 0.8, [], 1),
                AcronymSense("GPU", "General Purpose Unit", "gpu|general", 0.7, [], 1),
            ]
        }
        t1.sense_index = {x.sense_id: x for xs in t1.senses_by_acronym.values() for x in xs}
        t1.ranked = [
            Tier1OccurrenceRanking(
                occ=OccurrenceLite("GPU", 0, 3),
                candidate_scores={"gpu|graphics": 0.9, "gpu|general": 0.1},
                chosen_sense_id="gpu|graphics",
                gap=0.8,
                margin=0.88,
            )
        ]

        f.st_tiers_select_and_assemble(s, margin_threshold=0.2)

        assert s.extr is not None
        assert s.extr.resolutions[0].chosen_sense_id == "gpu|graphics"
        assert s.extr.resolutions[0].candidate_scores == {"gpu|graphics": 0.9, "gpu|general": 0.1}

class TestStTier2SelectAndAssemble:
    def test_flow_runs_with_tier2_enabled(self, _patch):

        ext_cfg = replace(ExtractionConfig(), tier2=Tier2Config(enabled=True, model_name="fake"))
        flow = ExtractionFlow(ext_cfg=ext_cfg, disambig_margin_threshold=0.99)  # encourage undecided in Tier-1

        _patch(flow.run,embed_texts=lambda *a, **k: None)

        text = "Graphics Processing Unit (GPU) does X. General Purpose Unit (GPU) does Y. Later, GPU appears."
        det_res, extr, reports = flow.run(text)

        assert extr is not None
        assert reports  # chain executed
