from dataclasses import replace
from typing import Literal

import numpy as np
from document_resolution.nlp.common.types import AcronymDetectorConfig, AcronymMeaning, OccurrenceLite
from document_resolution.nlp.extraction.acronyms.config import ExtractionConfig
from document_resolution.nlp.extraction.acronyms.engine import stage_funcs as f
from document_resolution.nlp.extraction.acronyms.engine.extract_flow import ExtractionFlow
from document_resolution.nlp.extraction.acronyms.engine.state import FlowState
from document_resolution.nlp.extraction.tiers import tier_2 as Tier2
from document_resolution.nlp.extraction.tiers.config import Tier2Config
from document_resolution.nlp.extraction.tiers.types import Tier1OccurrenceRanking


def _mk_state(*, mode: Literal["off", "auto", "on"]) -> FlowState:
    ext_cfg = replace(ExtractionConfig(), tier2=Tier2Config(mode=mode, weight=0.5, model_name="fake"))
    s = FlowState(text="ctx ... GPU ... kernel ...", det_cfg=AcronymDetectorConfig(), ext_cfg=ext_cfg)
    s.det_res = object()  # only asserted as not None in these stages
    return s


class TestSrTier2SemanticRerank:
    def test_tier2_disabled_sets_report_and_no_rankings(self, _patch):
        s = _mk_state(mode="off")

        # Set minimal Tier-1 ranked list (so "skipped=n" is meaningful)
        t1 = s.tier_1
        t1.ranked = [
            Tier1OccurrenceRanking(
                occ=OccurrenceLite("GPU", 5, 8),
                candidate_scores={"gpu|graphics": 0.5, "gpu|general": 0.49},
                chosen_meaning_id=None,
                gap=0.01,
                margin=0.02,
            )
        ]

        _patch(Tier2.embed_for_tier2, embed_texts=lambda *a, **k: None)

        f.st_tier2_semantic_rerank(s, auto_margin_ceiling=0.02)

        assert s.tier_2.report is not None
        assert s.tier_2.report.applied == 0
        assert s.tier_2.report.reasons["disabled"] == 1
        assert s.tier_2.ranked == []

    def test_tier2_model_unavailable_falls_back_cleanly(self, _patch):
        s = _mk_state(mode="on")

        # Seed Tier-1 work
        t1 = s.tier_1
        t1.meaning_index = {
            "gpu|graphics": AcronymMeaning("GPU", "Graphics Processing Unit", "gpu|graphics", 0.8, [], 1),
            "gpu|general": AcronymMeaning("GPU", "General Purpose Unit", "gpu|general", 0.7, [], 1),
        }

        t1.meaning_by_acronym = {"GPU": list(t1.meaning_index.values())}

        t1.ranked = [
            Tier1OccurrenceRanking(
                occ=OccurrenceLite("GPU", 5, 8),
                candidate_scores={"gpu|graphics": 0.5, "gpu|general": 0.5},
                chosen_meaning_id=None,
                gap=0.0,
                margin=0.0,
            )
        ]

        # Force embedder failure
        _patch(f.st_tier2_semantic_rerank, embed_texts=lambda *a, **k: None)
        _patch(Tier2.embed_for_tier2, embed_texts=lambda *a, **k: None)

        f.st_tier2_semantic_rerank(s, auto_margin_ceiling=0)

        rep = s.tier_2.report
        assert rep is not None
        assert rep.applied == 0
        assert rep.reasons["model_unavailable"] == 1
        assert s.tier_2.ranked[0].applied is False
        assert s.tier_2.ranked[0].skip_reason == "pending"

    def test_tier2_applies_and_blends_in_tier1_order(self, monkeypatch):
        s = _mk_state(mode="on")
        s.text = "kernel launch overhead ... GPU ..."

        t1 = s.tier_1
        t1.meaning_index = {
            "gpu|graphics": AcronymMeaning("GPU", "Graphics Processing Unit", "gpu|graphics", 0.8, [], 1),
            "gpu|general": AcronymMeaning("GPU", "General Purpose Unit", "gpu|general", 0.7, [], 1),
        }
        t1.meaning_by_acronym = {"GPU": list(t1.meaning_index.values())}
        t1.ranked = [
            Tier1OccurrenceRanking(
                occ=OccurrenceLite("GPU", 10, 13),
                candidate_scores={"gpu|graphics": 0.40, "gpu|general": 0.60},
                chosen_meaning_id=None,
                gap=0.20,
                margin=0.0,
            )
        ]

        def fake_embed_texts(texts, *, model=None, model_name=None):
            vecs = []
            for t in texts:
                if "Graphics Processing Unit" in t:
                    vecs.append([1.0, 0.0])
                elif "General Purpose Unit" in t:
                    vecs.append([0.0, 1.0])
                else:
                    # context -> align with graphics
                    vecs.append([1.0, 0.0])
            return np.asarray(vecs, dtype=np.float32)

        monkeypatch.setattr(Tier2, "embed_texts", fake_embed_texts, raising=True)

        f.st_tier2_semantic_rerank(s, auto_margin_ceiling=0)

        r2 = s.tier_2.ranked[0]
        assert r2.applied is True
        assert r2.blended_scores is not None
        assert list(r2.blended_scores.keys()) == ["gpu|graphics", "gpu|general"]
        assert r2.blended_scores["gpu|graphics"] > r2.blended_scores["gpu|general"]


class TestStTier1SelectAndAssemble:
    def test_select_and_assemble_uses_tier1_when_tier2_absent(self):
        ext_cfg = replace(ExtractionConfig(), tier2=Tier2Config(mode="off"))
        s = FlowState(text="x", det_cfg=AcronymDetectorConfig(), ext_cfg=ext_cfg)
        s.det_res = object()

        t1 = s.tier_1
        t1.meaning_by_acronym = {
            "GPU": [
                AcronymMeaning("GPU", "Graphics Processing Unit", "gpu|graphics", 0.8, [], 1),
                AcronymMeaning("GPU", "General Purpose Unit", "gpu|general", 0.7, [], 1),
            ]
        }
        t1.meaning_index = {x.meaning_id: x for xs in t1.meaning_by_acronym.values() for x in xs}
        t1.ranked = [
            Tier1OccurrenceRanking(
                occ=OccurrenceLite("GPU", 0, 3),
                candidate_scores={"gpu|graphics": 0.9, "gpu|general": 0.1},
                chosen_meaning_id="gpu|graphics",
                gap=0.8,
                margin=0.88,
            )
        ]

        f.st_tiers_select_and_assemble(s, margin_threshold=0.2)

        assert s.extr is not None
        assert s.extr.resolutions[0].chosen_meaning_id == "gpu|graphics"
        assert s.extr.resolutions[0].candidate_scores == {"gpu|graphics": 0.9, "gpu|general": 0.1}


class TestStTier2SelectAndAssemble:
    def test_flow_runs_with_tier2_enabled(self, monkeypatch):
        ext_cfg = replace(ExtractionConfig(), tier2=Tier2Config(mode="on", model_name="fake"))
        flow = ExtractionFlow(ext_cfg=ext_cfg, disambig_margin_threshold=0.99)

        text = "Graphics Processing Unit (GPU) does X. General Purpose Unit (GPU) does Y. Later, GPU appears."
        det_res, extr, reports = flow.run(text)

        assert extr is not None
        assert reports
