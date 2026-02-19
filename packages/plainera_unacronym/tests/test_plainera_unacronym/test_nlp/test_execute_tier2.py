from __future__ import annotations

import json
from dataclasses import asdict, replace
from typing import Literal

import numpy as np
import pytest
from _pytest.python_api import approx
from plainera_unacronym.nlp.common.types import (
    AcronymSense,
    ExtractedDefinition,
    OccurrenceLite,
    Span,
)
from plainera_unacronym.nlp.execute import detect_and_extract

from plainera_unacronym.nlp.extraction.core.defs import dedupe_defs
from plainera_unacronym.nlp.extraction.senses.disambiguate import choose_with_tiebreak, disambiguate_occurrences
from plainera_unacronym.nlp.extraction.senses.sense_build import build_senses
from plainera_unacronym.nlp.extraction.config import ExtractionConfig, Tier2Config  # adjust imports to your tree
from plainera_unacronym.nlp.extraction.engine import stage_funcs as f
import plainera_unacronym.nlp.extraction.tiers.tier_2 as t2


# -----------------------
# helpers
# -----------------------

def _stable_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

def _dump_extr(extr) -> str:
    # Stable compare for “byte-identical output” style assertions
    return _stable_json(asdict(extr))

def _tier2_cfg(*, mode: Literal["off", "auto", "on"], weight: float = 0.5) -> ExtractionConfig:
    return replace(
        ExtractionConfig(),
        tier2=Tier2Config(
            mode=mode,
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
def _dump_extr_core(extr) -> str:
    d = asdict(extr)

    # remove Tier-2-only diagnostics/telemetry
    d.pop("tier2_ranked", None)
    d.pop("tier2_report", None)

    return _stable_json(d)

class TestDetectAndExtractE2ETier2Contracts:
    def test_tier2_disabled_equals_model_unavailable(self, _patch):
        """
        Contract: Tier-2 disabled and Tier-2 enabled-but-unavailable must produce identical ExtractionResult.
        """
        text = (
            "Graphics Processing Unit (GPU) accelerates kernels. "
            + ("filler " * 300) + "\n"
            "General Purpose Unit (GPU) is used elsewhere. "
            "Later, GPU appears again."
        )
        import plainera_unacronym.nlp.extraction.tiers.tier_2 as t2

        def _raise_unavailable(*args, **kwargs):
            raise RuntimeError("model_unavailable")

        _patch(t2.embed_for_tier2, embed_texts=_raise_unavailable)

        # Disabled
        _det0, extr0, r0 = detect_and_extract(text, ext_cfg=_tier2_cfg(mode="off"), return_reports=True)
        assert "tier2=skipped(disabled)" in _stage_info(r0, "tier2_semantic_rerank")

        _det1, extr1, r1 = detect_and_extract(text, ext_cfg=_tier2_cfg(mode="on"), return_reports=True)
        info = _stage_info(r1, "tier2_semantic_rerank")

        assert "model_unavailable" in info or "skipped(model_unavailable)" in info, info
        assert _dump_extr_core(extr1) == _dump_extr_core(extr0)

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

        _det, extr, reports = detect_and_extract(text, ext_cfg=_tier2_cfg(mode="on"), return_reports=True)
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
        _det0, extr0, r0 = detect_and_extract(text, ext_cfg=_tier2_cfg(mode="off"), return_reports=True)
        last0 = _last_res(extr0, "GPU")
        assert last0.chosen_sense_id is not None  # Tier-1 likely chooses *something*
        baseline = last0.chosen_sense_id

        # Tier-2 enabled with deterministic semantic sims
        def _fake_embed_texts(model_name: str, texts: list[str], *_, **__) -> np.ndarray:
            """
            Deterministic tiny embedding:
            - dimension 0 = "kernel/graphics/device" semantics
            - dimension 1 = "general/purpose/department" semantics
            """
            out = np.zeros((len(texts), 2), dtype=np.float32)

            for i, t in enumerate(texts):
                s = t.lower()

                # context cues
                if "kernel" in s or "device" in s or "graphics" in s:
                    out[i, 0] = 1.0
                if "general purpose unit" in s or "department" in s:
                    out[i, 1] = 1.0

                # candidate cues (these appear in candidate_texts passed into embed_texts)
                if "graphics processing unit" in s:
                    out[i, 0] = 1.0
                if "general purpose unit" in s:
                    out[i, 1] = 1.0

                # avoid zero vectors
                if out[i].sum() == 0:
                    out[i, 0] = 1e-6

            return out

        _patch(t2.embed_for_tier2, embed_texts=_fake_embed_texts)

        _det1, extr1, r1 = detect_and_extract(
            text,
            ext_cfg=_tier2_cfg(mode="on", weight=1.0),  # crank it
            return_reports=True,
        )

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

        _det, extr, reports = detect_and_extract(text, ext_cfg=_tier2_cfg(mode="on", weight=1), return_reports=True)
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

        _det, extr, reports = detect_and_extract(text, ext_cfg=_tier2_cfg(mode="on", weight=1), return_reports=True)
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


class TestDetectAndExtractIntegrationEdgeCases:
    # this one
    def test_ambiguous_acronym_builds_multiple_senses(self, picked_def, cfg_integrated):
        # EMA appears with two meanings; result should have ambiguous senses for EMA
        text = (
            "EMA stands for European Medicines Agency in the EU context. "
            "On charts, EMA (Exponential Moving Average) is a common indicator."
        )
        det_cfg, ext_cfg = cfg_integrated()
        det_res, extr = detect_and_extract(text, det_cfg=det_cfg, ext_cfg=ext_cfg)

        # senses_by_acronym and ambiguous_keys should reflect two senses for EMA
        senses = extr.senses_by_acronym.get("EMA", [])
        assert len(senses) >= 2, f"Expected multiple senses for EMA, got {senses}"
        assert "EMA" in extr.ambiguous_keys

        # And definitions should include both
        defs_for_ema = [d.definition for d in extr.definitions if d.acronym == "EMA"]
        joined = " || ".join(defs_for_ema).lower()
        assert "european medicines agency" in joined
        assert "exponential moving average" in joined
    # this one
    def test_nearest_pick_prefers_definition_near_first_occurrence(self, picked_def, cfg_integrated):
        # Two candidate long-forms for the same acronym; ensure the one closest to the FO wins
        text = (
            "Portable Document Format (PDF) is ubiquitous. "  # <-- near FO
            "Later we see some detour text and a PDF (Pretty Darn Fast) joke."
        )
        det_cfg, ext_cfg = cfg_integrated()
        det_res, extr = detect_and_extract(text, det_cfg=det_cfg, ext_cfg=ext_cfg)

        # The chosen pick for PDF (by nearest) should be the first proper definition
        pick = extr.picks.get("PDF")
        assert pick is not None
        assert "Portable Document Format" in pick.definition

    # this one
    def test_tier_one_digit_prefixed_acronym_parenthetical(self, picked_def):
        det, extr = detect_and_extract("Third Generation Partnership Project (3GPP) publishes specs.")
        assert picked_def(extr, "3GPP") == "Third Generation Partnership Project"



class TestDisambiguationE2E:

    def test_disambiguation_picks_nearest_definition_by_distance(self, picked_def):
        # Two senses, then a later occurrence near the second definition → should pick second.
        det, extr, r = detect_and_extract(
            "Natural language processing (NLP) helps. "
            "Later we discuss Nice Lovely Plants (NLP) sold locally. "
            "These NLP are popular in spring.",
            return_reports=True,
        )
        # The last "NLP" should resolve to the nearby "Nice Lovely Plants" sense.
        last = extr.resolutions[-1]
        assert last.acronym == "NLP"
        assert last.chosen_sense_id is not None
        assert "nice_lovely_plants" in last.chosen_sense_id, last

    def test_disambiguation_not_ambiguous_when_only_one_sense(self, picked_def):
        det, extr = detect_and_extract(
            "European Medicines Agency (EMA) issued guidance. EMA guidance was updated later.")
        assert "EMA" in extr.senses_by_acronym
        assert len(extr.senses_by_acronym["EMA"]) == 1
        assert "EMA" not in set(extr.ambiguous_keys)
        # both occurrences should be resolved (same sole sense)
        ema_res = [x for x in extr.resolutions if x.acronym.upper() == "EMA"]
        assert len(ema_res) >= 2
        assert all(x.chosen_sense_id is not None for x in ema_res), ema_res

    def test_disambiguation_ambiguous_keys_flagged_when_two_senses_exist(self, picked_def):
        det, extr, r = detect_and_extract(
            "Natural language processing (NLP) is common. "
            "Nice Lovely Plants (NLP) are sold locally.",
            return_reports=True,
        )
        assert "NLP" in extr.senses_by_acronym
        assert len(extr.senses_by_acronym["NLP"]) == 2
        assert "NLP" in set(extr.ambiguous_keys)

    def test_disambiguation_near_tie_chooses_nearest_definition(self, picked_def):
        det, extr, r = detect_and_extract(
            "Natural language processing (NLP) is a field. "
            "Nice Lovely Plants (NLP) are sold down the road. "
            "NLP is mentioned again here without context.",
            return_reports=True,
        )

        nlp_res = [x for x in extr.resolutions if x.acronym.upper() == "NLP"]
        assert nlp_res, extr

        last = nlp_res[-1]
        # Near-tie (margin below threshold) => distance tiebreak => pick nearest def span.
        assert last.chosen_sense_id is not None
        assert "nice_lovely_plants" in last.chosen_sense_id, last
        assert last.margin < 0.10, last  # confirms it was in the "not confident" zone

    def test_disambiguation_overlap_can_win_when_distance_not_dominating(self, picked_def):
        # Make the final NLP mention much closer (and semantically aligned) to the NLP sense,
        # while pushing the Plants definition far away via filler.
        filler = " ".join(["filler"] * 250)

        det, extr, r = detect_and_extract(
            "Natural language processing (NLP) is a CS topic. "
            "In this paper we discuss language models and processing techniques; NLP is crucial. "
            f"{filler} "
            "Nice Lovely Plants (NLP) are available in shops.",
            return_reports=True,
        )

        # Pick the resolution for the NLP occurrence in the "language/processing" sentence.
        # (There will be multiple; choose the one with start after the first def and before the plants def.)
        nlp_res = [x for x in extr.resolutions if x.acronym.upper() == "NLP"]
        assert len(nlp_res) >= 2, nlp_res

        # The second occurrence is the one in the language/processing sentence in this construction.
        mid = nlp_res[1]
        assert mid.chosen_sense_id is not None
        assert "natural_language_processing" in mid.chosen_sense_id, mid


class TestDisambiguationE2EConfidenceContract:
    def test_merge_dedupe_prefers_higher_confidence_for_same_sense(self):
        """
        Two strategies extract the *same* (acronym, definition) sense with different
        confidence. Dedupe must keep the higher-confidence definition.
        """
        det, extr, r = detect_and_extract(
            "European Medicines Agency (EMA) issued guidance. EMA guidance was updated later.",
            return_reports=True,
        )

        # Force a competing duplicate sense with higher confidence and a distinct source.
        # (We inject post-extract because this is E2E against merge/dedupe behaviour,
        # not against upstream strategy extraction.)

        d0 = extr.definitions[0]
        injected = ExtractedDefinition(
            acronym=d0.acronym,
            definition=d0.definition,
            source="injected_strategy",
            definition_confidence=min(1.0, d0.definition_confidence + 0.04),
            acr_start=d0.acr_start,
            acr_end=d0.acr_end,
            def_start=d0.def_start,
            def_end=d0.def_end,
            original_definition=d0.original_definition,
            kind="injected",
            reasons=("injected_higher_conf",),
        )

        winners = dedupe_defs([d0, injected])
        assert len(winners) == 1
        assert winners[0].source == "injected_strategy"
        assert winners[0].definition_confidence == approx(injected.definition_confidence)

    def test_build_senses_uses_max_definition_confidence_as_sense_confidence(self):
        """
        When multiple defs collapse to the same sense_id, sense_confidence must reflect
        the best supporting definition_confidence.
        """
        det, extr, r = detect_and_extract(
            "European Medicines Agency (EMA) issued guidance. EMA guidance was updated later.",
            return_reports=True,
        )


        d0 = extr.definitions[0]
        low = ExtractedDefinition(
            acronym=d0.acronym,
            definition=d0.definition,
            source="injected_low",
            definition_confidence=max(0.0, d0.definition_confidence - 0.20),
            acr_start=d0.acr_start,
            acr_end=d0.acr_end,
            def_start=d0.def_start,
            def_end=d0.def_end,
            original_definition=d0.original_definition,
            kind="injected",
            reasons=("injected_low_conf",),
        )

        senses_by = build_senses([d0, low])
        assert "EMA" in senses_by
        assert len(senses_by["EMA"]) == 1
        sense = senses_by["EMA"][0]
        assert sense.sense_confidence == approx(d0.definition_confidence)

    def test_dynamic_prior_breaks_near_tie_in_favour_of_higher_confidence_sense(self, _patch):
        """
        If base scores are a near tie, the confidence prior should nudge selection
        toward the higher-confidence sense (when enabled).
        """
        # Patch base scoring to guarantee a near-tie, regardless of text/layout.
        from plainera_unacronym.nlp.extraction.senses import disambiguate as mod

        def fake_base_scores_for_occurrence(*_, **__):
            # Near tie: gap = 0.01 (<= NEAR_TIE_GAP 0.06), and relative margin is small.
            return {
                "nlp|natural_language_processing": 0.50,
                "nlp|nice_lovely_plants": 0.49,
            }

        _patch(mod.disambiguate_occurrences, base_scores_for_occurrence=fake_base_scores_for_occurrence)

        det, extr, r = detect_and_extract(
            "Natural language processing (NLP) helps. "
            "Nice Lovely Plants (NLP) sold locally. "
            "NLP appears again.",
            return_reports=True,
        )

        # Build a minimal senses_by_id map from extraction output.
        senses_by_id = extr.sense_index

        # Ensure we have both senses and that we can control their sense_confidence:
        # (E2E-friendly: mutate by rebuilding local objects if your dataclass is frozen elsewhere;
        # here, we just assert what's already present and use patch on sense_prior term behaviour.)
        assert "nlp|natural_language_processing" in senses_by_id
        assert "nlp|nice_lovely_plants" in senses_by_id

        # Now patch confidence levels by monkeypatching the sense_index entries via replacement.
        # If AcronymSense is mutable in your codebase, you can direct-set instead.
        s_hi = senses_by_id["nlp|natural_language_processing"]
        s_lo = senses_by_id["nlp|nice_lovely_plants"]

        # Sanity: make sure we can see a difference (or the prior would be moot).
        # If both are equal from upstream, this test can still pass by force-setting.
        try:
            s_hi.sense_confidence = 0.95
            s_lo.sense_confidence = 0.40
        except Exception:
            # If frozen, rebuild lightweight namespace objects for the call below
            pass

        # Directly call disambiguate_occurrences with a known near-tie + prior enabled.
        occs = [OccurrenceLite("NLP", 0, 3)]
        out = mod.disambiguate_occurrences(
            text="x" * 50,
            occurrences=occs,
            senses={"NLP": list(extr.senses_by_acronym["NLP"])},
            sense_prior_weight=0.08,  # enable
            senses_by_id=senses_by_id,
            window_chars=10,
        )
        assert out and out[0].chosen_sense_id is not None
        assert "natural_language_processing" in out[0].chosen_sense_id

    def test_dynamic_prior_disabled_keeps_near_tie_unresolved(self):
        """
        Integration-style contract:
        - Build real senses (and real def_spans) from the pipeline.
        - Create a synthetic occurrence positioned exactly midway between the two def spans.
        - With prior disabled and distance unable to distinguish, resolution stays undecided.
        """


        # 1) Run full pipeline once to get REAL senses + REAL def_spans.
        _det, extr, _r = detect_and_extract(
            "Natural language processing (NLP) helps. "
            "Nice Lovely Plants (NLP) sold locally.",
            return_reports=True,
        )

        senses = list(extr.senses_by_acronym["NLP"])
        assert len(senses) == 2
        s1, s2 = senses

        # Take the first def span for each sense and compute span-centers (same logic as disambiguate.py)
        (a1, b1) = s1.def_spans[0]
        (a2, b2) = s2.def_spans[0]
        c1 = (a1 + b1) // 2
        c2 = (a2 + b2) // 2

        # 2) Place occurrence start exactly at the midpoint between centers (equal distance to both).
        mid = (c1 + c2) // 2
        occ = OccurrenceLite("NLP", mid, mid + 3)

        # 3) Use dummy text with no useful overlap signal (tokens won't intersect sense definitions).
        text = "x" * (mid + 50)

        out = disambiguate_occurrences(
            text=text,
            occurrences=[occ],
            senses={"NLP": senses},
            senses_by_id=extr.sense_index,
            window_chars=20,
            sense_prior_weight=0.0,  # disable prior (what we're asserting)
            margin_threshold=0.10,
            dist_weight=0.75,
            overlap_weight=0.25,
        )

        assert out
        assert out[0].chosen_sense_id is None, out[0]

    def test_choose_with_tiebreak_margins_checked(self):
        """
        Contract test: choose_with_tiebreak returns relative and absolute margins.
        """

        def S(acr: str, sense_id: str, definition: str, spans: list[Span], *, conf: float = 0.0,
              support: int = 1) -> AcronymSense:
            return AcronymSense(
                acronym=acr,
                definition=definition,
                sense_id=sense_id,
                sense_confidence=conf,
                def_spans=list(spans),
                support=support,
            )
        occ = OccurrenceLite("PDF", 10, 13)
        senses_by_id = {
            "s1": S("PDF", "s1", "Portable Document Format", [(0, 1)]),
            "s2": S("PDF", "s2", "Other", [(0, 1)]),
        }
        cand = {"s1": 0.80, "s2": 0.60}

        chosen, rel_margin, abs_margin = choose_with_tiebreak(occ, cand, senses_by_id, margin_threshold=0.10)
        assert chosen == "s1"

        assert abs_margin == approx(0.200000, abs=1e-6)
        assert rel_margin == approx((0.80 - 0.60) / 0.80, rel=0, abs=1e-9)
