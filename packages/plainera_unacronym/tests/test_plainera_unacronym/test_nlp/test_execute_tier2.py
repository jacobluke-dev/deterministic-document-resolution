from __future__ import annotations

from dataclasses import asdict, replace
import json
import pytest

from plainera_unacronym.nlp.execute import detect_and_extract
from plainera_unacronym.nlp.extraction.config import ExtractionConfig, Tier2Config  # adjust imports to your tree
from plainera_unacronym.nlp.extraction.engine import stage_funcs as f

# -----------------------
# helpers
# -----------------------

def _stable_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

def _dump_extr(extr) -> str:
    # Stable compare for “byte-identical output” style assertions
    return _stable_json(asdict(extr))

def _tier2_cfg(*, enabled: bool, weight: float = 0.5) -> ExtractionConfig:
    return replace(
        ExtractionConfig(),
        tier2=Tier2Config(
            enabled=enabled,
            model_name="fake-model",
            weight=weight,
            # only_when_undecided=True,  # if you have it
        ),
    )

def _stage_info(reports, name: str) -> str:
    return next(r for r in reports if r.name == name).info

def _last_res(extr, acr: str):
    xs = [r for r in extr.resolutions if r.acronym.upper() == acr.upper()]
    assert xs, f"no resolutions for {acr}"
    return xs[-1]


# -----------------------
# Tier-2 E2E: contracts
# -----------------------

class TestDetectAndExtractE2ETier2Contracts:
    def test_tier2_disabled_equals_model_unavailable(self, _patch):
        """
        Contract: Tier-2 disabled and Tier-2 enabled-but-unavailable must produce identical ExtractionResult.
        """
        text = (
            "Graphics Processing Unit (GPU) accelerates kernels. "
            "General Purpose Unit (GPU) is used elsewhere. "
            "Later, GPU appears again."
        )

        # Disabled
        _det0, extr0, r0 = detect_and_extract(text, ext_cfg=_tier2_cfg(enabled=False), return_reports=True)
        assert "tier2=skipped(disabled)" in _stage_info(r0, "tier2_semantic_rerank")

        # Enabled but semantic helper returns None => model_unavailable
        def fake_sims_for_context_and_candidates(*, model_name, context, candidate_texts):
            return None

        _patch(f.st_tier2_semantic_rerank, sims_for_context_and_candidates=fake_sims_for_context_and_candidates)

        _det1, extr1, r1 = detect_and_extract(text, ext_cfg=_tier2_cfg(enabled=True), return_reports=True)
        info = _stage_info(r1, "tier2_semantic_rerank")
        assert "model_unavailable" in info or "applied(0)" in info, info

        assert _dump_extr(extr1) == _dump_extr(extr0)

    def test_tier2_skips_single_candidate_everywhere(self, _patch):
        """
        Contract: if an acronym has only one sense, Tier-2 must not apply.
        """
        text = "European Medicines Agency (EMA) issued guidance. EMA guidance was updated."

        # Make Tier-2 enabled, but it should never apply (single candidate in Tier-1)
        def fake_sims_for_context_and_candidates(*, model_name, context, candidate_texts):
            # Would apply if called; we expect it not to be called.
            raise AssertionError("Tier-2 sims should not be requested for single-candidate cases")

        _patch(f.st_tier2_semantic_rerank, sims_for_context_and_candidates=fake_sims_for_context_and_candidates)

        _det, extr, reports = detect_and_extract(text, ext_cfg=_tier2_cfg(enabled=True), return_reports=True)
        info = _stage_info(reports, "tier2_semantic_rerank")
        # Your info string may differ; assert the intent:
        assert "applied(0)" in info or "single_candidate" in info, info
        # sanity: only one sense
        assert len(extr.senses_by_acronym["EMA"]) == 1


# -----------------------
# Tier-2 E2E: “wins”
# -----------------------

class TestDetectAndExtractE2ETier2AcronymWins:
    def test_tier2_can_override_distance_when_semantics_strong_gpu(self, _patch):
        """
        Scenario:
          - Two GPU senses defined.
          - Final occurrence is closer to the *wrong* definition (distance misleads Tier-1).
          - Tier-2 semantics should rerank/blend so the correct sense wins.

        We patch sims to be “context aware” via keyword triggers.
        """
        text = (
            "Graphics Processing Unit (GPU) accelerates kernel execution on the device. "
            "Some filler words to push distance around. " * 20 +
            "General Purpose Unit (GPU) is used in a different department. "
            "The GPU was saturated due to kernel launch overhead."
        )

        # Baseline: Tier-2 disabled (records what Tier-1 does)
        _det0, extr0, r0 = detect_and_extract(text, ext_cfg=_tier2_cfg(enabled=False), return_reports=True)
        last0 = _last_res(extr0, "GPU")
        assert last0.chosen_sense_id is not None  # Tier-1 likely chooses *something*
        baseline = last0.chosen_sense_id

        # Tier-2 enabled with deterministic semantic sims
        def fake_sims_for_context_and_candidates(*, model_name, context, candidate_texts):
            ctx = context.lower()
            out = []
            for t in candidate_texts:
                tt = t.lower()
                if "kernel" in ctx and "graphics processing unit" in tt:
                    out.append(0.99)
                elif "kernel" in ctx and "general purpose unit" in tt:
                    out.append(0.01)
                else:
                    out.append(0.50)  # neutral
            return out

        _patch(f.st_tier2_semantic_rerank, sims_for_context_and_candidates=fake_sims_for_context_and_candidates)

        _det1, extr1, r1 = detect_and_extract(text, ext_cfg=_tier2_cfg(enabled=True, weight=0.6), return_reports=True)
        info = _stage_info(r1, "tier2_semantic_rerank")
        assert "applied(" in info, info

        last1 = _last_res(extr1, "GPU")
        assert last1.chosen_sense_id is not None

        # Tier-2 should pull toward “graphics_processing_unit”
        assert "graphics_processing_unit" in last1.chosen_sense_id, last1

        # Candidate set invariant: keys unchanged (only scores/order change)
        assert set(last1.candidate_scores.keys()) == set(last0.candidate_scores.keys())

        # And it should differ from baseline if baseline was the wrong one (not guaranteed, but usually true)
        # If this is flaky, keep only the positive assertion above.
        assert last1.chosen_sense_id != baseline or "graphics_processing_unit" in baseline

    def test_tier2_api_programming_interface_vs_active_pharmaceutical_ingredient(self, _patch):
        text = (
            "Application Programming Interface (API) defines endpoints and contracts. "
            "Some filler words. " * 15 +
            "Active Pharmaceutical Ingredient (API) must be controlled under GMP. "
            "Our API exposes a REST endpoint for searching."
        )

        def fake_sims_for_context_and_candidates(*, model_name, context, candidate_texts):
            ctx = context.lower()
            out = []
            for t in candidate_texts:
                tt = t.lower()
                if ("rest" in ctx or "endpoint" in ctx) and "application programming interface" in tt:
                    out.append(0.95)
                elif ("gmp" in ctx or "pharmaceutical" in ctx) and "active pharmaceutical ingredient" in tt:
                    out.append(0.95)
                else:
                    out.append(0.10)
            return out

        _patch(f.st_tier2_semantic_rerank, sims_for_context_and_candidates=fake_sims_for_context_and_candidates)

        _det, extr, reports = detect_and_extract(text, ext_cfg=_tier2_cfg(enabled=True, weight=0.7), return_reports=True)
        last = _last_res(extr, "API")
        assert last.chosen_sense_id is not None
        assert "application_programming_interface" in last.chosen_sense_id, last
        assert "applied(" in _stage_info(reports, "tier2_semantic_rerank")

    def test_tier2_nhs_health_service_vs_honour_society(self, _patch):
        text = (
            "National Health Service (NHS) publishes guidance for hospitals. "
            "Some filler. " * 10 +
            "National Honor Society (NHS) recognises student achievement in the US. "
            "The NHS hospital trust updated policy."
        )

        def fake_sims_for_context_and_candidates(*, model_name, context, candidate_texts):
            ctx = context.lower()
            out = []
            for t in candidate_texts:
                tt = t.lower()
                if ("hospital" in ctx or "trust" in ctx) and "national health service" in tt:
                    out.append(0.97)
                elif ("student" in ctx or "achievement" in ctx) and "national honor society" in tt:
                    out.append(0.97)
                else:
                    out.append(0.20)
            return out

        _patch(f.st_tier2_semantic_rerank, sims_for_context_and_candidates=fake_sims_for_context_and_candidates)

        _det, extr, reports = detect_and_extract(text, ext_cfg=_tier2_cfg(enabled=True, weight=0.7), return_reports=True)
        last = _last_res(extr, "NHS")
        assert last.chosen_sense_id is not None
        assert "national_health_service" in last.chosen_sense_id, last
        assert "applied(" in _stage_info(reports, "tier2_semantic_rerank")



class TestDetectAndExtractE2ETier2MixedCaseAcronyms:

    def test_stylised_ios_parenthetical(self, picked_def):
        det, extr = detect_and_extract("iOS (iPhone Operating System) is supported.")
        assert picked_def(extr, "iOS") in {"iPhone Operating System"}, extr.picks.get("iOS")

    def test_stylised_ebay_parenthetical(self, picked_def):
        det, extr = detect_and_extract("eBay (electronic Bay) is a marketplace.")
        assert picked_def(extr, "eBay") in {"electronic Bay"}, extr.picks.get("eBay")

    def test_stylised_latex_parenthetical(self, picked_def):
        det, extr = detect_and_extract("LaTeX (Lamport TeX) is used for typesetting.")
        assert picked_def(extr, "LaTeX") in {"Lamport TeX"}, extr.picks.get("LaTeX")

    def test_stylised_latex_parenthetical_inverse(self, picked_def):
        det, extr = detect_and_extract("Lamport TeX (LaTeX) is used for typesetting.")
        assert picked_def(extr, "LaTeX") in {"LaTeX"}, extr.picks.get("LaTeX")
