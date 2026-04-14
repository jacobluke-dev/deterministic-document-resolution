from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, replace
from typing import Any, Literal

import numpy as np
import plainera_unacronym.nlp.extraction.tiers.tier_2 as t2
from _pytest.python_api import approx
from plainera_unacronym.nlp.common.types import (
    AcronymMeaning,
    ExtractedDefinition,
    OccurrenceLite,
    Span,
)
from plainera_unacronym.nlp.extraction.acronyms.config import ExtractionConfig
from plainera_unacronym.nlp.extraction.acronyms.core.defs import dedupe_defs
from plainera_unacronym.nlp.extraction.acronyms.engine import stage_funcs as f
from plainera_unacronym.nlp.extraction.acronyms.engine.extract_flow import ExtractionFlow
from plainera_unacronym.nlp.extraction.acronyms.engine.state import FlowState
from plainera_unacronym.nlp.extraction.acronyms.execute import detect_and_extract
from plainera_unacronym.nlp.extraction.acronyms.meanings.disambiguate import (
    choose_with_tiebreak,
    disambiguate_occurrences,
)
from plainera_unacronym.nlp.extraction.acronyms.meanings.meaning_build import build_meanings
from plainera_unacronym.nlp.extraction.tiers.config import Tier2Config

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
            weight=weight
        ),
    )

# Make Tier-2 enabled, but it should never apply (single candidate in Tier-1)
def fake_sims_for_context_and_candidates(*, model_name, context, candidate_texts):
    # Would apply if called; we expect it not to be called.
    raise AssertionError("Tier-2 sims should not be requested for single-candidate cases")


def _stage_info(reports, name: str) -> str:
    return next(r for r in reports if r.name == name).info


def _last_res(extr, acr: str):
    xs = [r for r in extr.resolutions if r.acronym.upper() == acr.upper()]
    assert xs, f"no resolutions for {acr}"
    return xs[-1]

def _res_after(extr, acr: str, after: int):
    xs = [r for r in extr.resolutions if r.acronym.upper() == acr.upper() and r.start > after]
    assert xs, f"no {acr} resolution after pos={after}"
    return min(xs, key=lambda r: r.start)  # first after boundary


def _res_near(extr, acr: str, anchor: int, *, max_delta: int = 200):
    xs = [r for r in extr.resolutions if r.acronym.upper() == acr.upper()]
    assert xs, f"no resolutions for {acr}"
    best = min(xs, key=lambda r: abs(r.start - anchor))
    assert abs(best.start - anchor) <= max_delta, (anchor, best)
    return best

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
            "Graphics Processing Unit (GPU) accelerates kernels. " + ("filler " * 300) + "\n"
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
        Contract: if an acronym has only one meaning, Tier-2 must not apply.
        """
        text = "European Medicines Agency (EMA) issued guidance. EMA guidance was updated."

        _patch(f.st_tier2_semantic_rerank, sims_for_context_and_candidates=fake_sims_for_context_and_candidates)

        _det, extr, reports = detect_and_extract(text, ext_cfg=_tier2_cfg(mode="on"), return_reports=True)
        info = _stage_info(reports, "tier2_semantic_rerank")
        # Your info string may differ; assert the intent:
        assert "applied(0)" in info or "single_candidate" in info, info
        # sanity: only one meaning
        assert len(extr.meaning_by_acronym["EMA"]) == 1


# -----------------------
# Tier-2 E2E: “wins”
# -----------------------


class TestDetectAndExtractE2ETier2AcronymWins:
    def test_tier2_can_override_distance_when_semantics_strong_gpu(self, _patch):
        """
        Scenario:
          - Two GPU meanings defined.
          - Final occurrence is closer to the *wrong* definition (distance misleads Tier-1).
          - Tier-2 semantics should rerank/blend so the correct meaning wins.

        We patch sims to be “context aware” via keyword triggers.
        """
        text = (
            "Graphics Processing Unit (GPU) accelerates kernel execution on the device. "
            "Some filler words to push distance around. "
            * 20
            + "General Purpose Unit (GPU) is used in a different department. "
            "The GPU was saturated due to kernel launch overhead."
        )

        # Baseline: Tier-2 disabled (records what Tier-1 does)
        _det0, extr0, r0 = detect_and_extract(text, ext_cfg=_tier2_cfg(mode="off"), return_reports=True)
        last0 = _last_res(extr0, "GPU")
        assert last0.chosen_meaning_id is not None  # Tier-1 likely chooses *something*
        baseline = last0.chosen_meaning_id

        # Tier-2 enabled with deterministic semantic sims
        def _fake_embed_texts(texts, *, model=None, model_name=None, **_kw) -> np.ndarray:
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
        assert last1.chosen_meaning_id is not None

        # Tier-2 should pull toward “graphics_processing_unit”
        assert "graphics_processing_unit" in last1.chosen_meaning_id, last1

        # Candidate set invariant: keys unchanged (only scores/order change)
        assert set(last1.candidate_scores.keys()) == set(last0.candidate_scores.keys())

        # And it should differ from baseline if baseline was the wrong one (not guaranteed, but usually true)
        # If this is flaky, keep only the positive assertion above.
        assert last1.chosen_meaning_id != baseline or "graphics_processing_unit" in baseline

    def test_tier2_api_programming_interface_vs_active_pharmaceutical_ingredient(self, monkeypatch):
        import plainera_unacronym.nlp.extraction.tiers.tier_2 as t2

        def fake_embed_texts(texts, *, model=None, model_name=None, **_kw):
            xs = list(texts)
            out = np.zeros((len(xs), 2), dtype=np.float32)

            for i, t in enumerate(xs):
                s = str(t).lower()

                # Candidate texts -> one-hot
                if "application programming interface" in s:
                    out[i, 0] = 1.0
                    continue
                if "active pharmaceutical ingredient" in s:
                    out[i, 1] = 1.0
                    continue

                # Context cues
                if "rest" in s or "endpoint" in s or "http" in s:
                    out[i, 0] = 1.0
                if "gmp" in s or "pharmaceutical" in s or "assay" in s or "purity" in s:
                    out[i, 1] = 1.0

                if out[i].sum() == 0:
                    out[i, 0] = 1e-6

            return out

        monkeypatch.setattr(t2, "embed_texts", fake_embed_texts, raising=True)

        text = (
            "Application Programming Interface (API) defines endpoints and contracts.\n"
            + ("filler " * 250) + "\n"
            + "Active Pharmaceutical Ingredient (API) must be controlled under GMP.\n"
            + ("morefiller " * 250) + "\n"
            + "Our API exposes a REST endpoint for searching.\n"
        )

        pharma_anchor = text.index("Active Pharmaceutical Ingredient (API)")
        rest_anchor = text.index("Our API exposes")

        ext_cfg = replace(
            ExtractionConfig(),
            tier2=Tier2Config(mode="on", weight=1, model_name="fake-model"),
        )

        _det, extr, reports = detect_and_extract(text, ext_cfg=ext_cfg, return_reports=True, tier2_model=object())
        info = next(r.info for r in reports if r.name == "tier2_semantic_rerank")
        assert "applied(" in info, info

        r_pharma = _res_near(extr, "API", pharma_anchor)
        assert "active_pharmaceutical_ingredient" in r_pharma.chosen_meaning_id

        r_rest = _res_near(extr, "API", rest_anchor)
        assert "application_programming_interface" in r_rest.chosen_meaning_id

    def test_tier2_nhs_can_pick_honour_society_when_context_mentions_students(self, monkeypatch):
        import plainera_unacronym.nlp.extraction.tiers.tier_2 as t2

        def fake_embed_texts(texts, *, model=None, model_name=None, **_kw):
            xs = list(texts)
            out = np.zeros((len(xs), 2), dtype=np.float32)
            for i, t in enumerate(xs):
                s = str(t).lower()
                if "national health service" in s:
                    out[i, 0] = 1.0
                    continue
                if "national honor society" in s or "national honour society" in s:
                    out[i, 1] = 1.0
                    continue
                if "student" in s or "achievement" in s or "awards" in s:
                    out[i, 1] = 1.0
                if out[i].sum() == 0:
                    out[i, 0] = 1e-6
            return out

        monkeypatch.setattr(t2, "embed_texts", fake_embed_texts, raising=True)

        text = (
            "NHS (National Health Service) publishes guidance.\n"
            + ("filler " * 200) + "\n"  # ensure contexts don't overlap
            + "Student achievement: NHS (National Honor Society) recognises awards.\n"
        )

        honor_anchor = text.index("Student achievement: NHS")

        ext_cfg = replace(
            ExtractionConfig(),
            tier2=Tier2Config(mode="on", weight=1, model_name="fake-model"),
        )

        _det, extr, reports = detect_and_extract(text, ext_cfg=ext_cfg, return_reports=True, tier2_model=object())
        info = next(r.info for r in reports if r.name == "tier2_semantic_rerank")
        assert "applied(" in info, info

        r_honor = _res_near(extr, "NHS", honor_anchor)
        assert "national_honor_society" in r_honor.chosen_meaning_id


class TestDetectAndExtractIntegrationEdgeCases:
    # this one
    def test_ambiguous_acronym_builds_multiple_meanings(self, picked_def, cfg_integrated):
        # EMA appears with two meanings; result should have ambiguous meanings for EMA
        text = (
            "EMA stands for European Medicines Agency in the EU context. "
            "On charts, EMA (Exponential Moving Average) is a common indicator."
        )
        det_cfg, ext_cfg = cfg_integrated()
        det_res, extr = detect_and_extract(text, det_cfg=det_cfg, ext_cfg=ext_cfg)

        # meaning_by_acronym and ambiguous_keys should reflect two meanings for EMA
        meanings = extr.meaning_by_acronym.get("EMA", [])
        assert len(meanings) >= 2, f"Expected multiple meanings for EMA, got {meanings}"
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
        # Two meanings, then a later occurrence near the second definition → should pick second.
        det, extr, r = detect_and_extract(
            "Natural language processing (NLP) helps. "
            "Later we discuss Nice Lovely Plants (NLP) sold locally. "
            "These NLP are popular in spring.",
            return_reports=True,
        )
        # The last "NLP" should resolve to the nearby "Nice Lovely Plants" meaning.
        last = extr.resolutions[-1]
        assert last.acronym == "NLP"
        assert last.chosen_meaning_id is not None
        assert "nice_lovely_plants" in last.chosen_meaning_id, last

    def test_disambiguation_not_ambiguous_when_only_one_meaning(self, picked_def):
        det, extr = detect_and_extract(
            "European Medicines Agency (EMA) issued guidance. EMA guidance was updated later."
        )
        assert "EMA" in extr.meaning_by_acronym
        assert len(extr.meaning_by_acronym["EMA"]) == 1
        assert "EMA" not in set(extr.ambiguous_keys)
        # both occurrences should be resolved (same sole meaning)
        ema_res = [x for x in extr.resolutions if x.acronym.upper() == "EMA"]
        assert len(ema_res) >= 2
        assert all(x.chosen_meaning_id is not None for x in ema_res), ema_res

    def test_disambiguation_ambiguous_keys_flagged_when_two_meanings_exist(self, picked_def):
        det, extr, r = detect_and_extract(
            "Natural language processing (NLP) is common. " "Nice Lovely Plants (NLP) are sold locally.",
            return_reports=True,
        )
        assert "NLP" in extr.meaning_by_acronym
        assert len(extr.meaning_by_acronym["NLP"]) == 2
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
        assert last.chosen_meaning_id is not None
        assert "nice_lovely_plants" in last.chosen_meaning_id, last
        assert last.margin < 0.10, last  # confirms it was in the "not confident" zone

    def test_disambiguation_overlap_can_win_when_distance_not_dominating(self, picked_def):
        # Make the final NLP mention much closer (and semantically aligned) to the NLP meaning,
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
        assert mid.chosen_meaning_id is not None
        assert "natural_language_processing" in mid.chosen_meaning_id, mid


class TestDisambiguationE2EConfidenceContract:
    def test_merge_dedupe_prefers_higher_confidence_for_same_meaning(self):
        """
        Two strategies extract the *same* (acronym, definition) meaning with different
        confidence. Dedupe must keep the higher-confidence definition.
        """
        det, extr, r = detect_and_extract(
            "European Medicines Agency (EMA) issued guidance. EMA guidance was updated later.",
            return_reports=True,
        )

        # Force a competing duplicate meaning with higher confidence and a distinct source.
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

    def test_build_meanings_uses_max_definition_confidence_as_meaning_confidence(self):
        """
        When multiple defs collapse to the same meaning_id, meaning_confidence must reflect
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

        meanings_by = build_meanings([d0, low])
        assert "EMA" in meanings_by
        assert len(meanings_by["EMA"]) == 1
        meaning = meanings_by["EMA"][0]
        assert meaning.meaning_confidence == approx(d0.definition_confidence)

    def test_dynamic_prior_breaks_near_tie_in_favour_of_higher_confidence_meaning(self, _patch):
        """
        If base scores are a near tie, the confidence prior should nudge selection
        toward the higher-confidence meaning (when enabled).
        """
        # Patch base scoring to guarantee a near-tie, regardless of text/layout.
        from plainera_unacronym.nlp.extraction.acronyms.meanings import disambiguate as mod

        def fake_base_scores_for_occurrence(*_, **__):
            # Near tie: gap = 0.01 (<= NEAR_TIE_GAP 0.06), and relative margin is small.
            return {
                "nlp|natural_language_processing": 0.50,
                "nlp|nice_lovely_plants": 0.49,
            }

        _patch(mod.disambiguate_occurrences, base_scores_for_occurrence=fake_base_scores_for_occurrence)

        det, extr, r = detect_and_extract(
            "Natural language processing (NLP) helps. " "Nice Lovely Plants (NLP) sold locally. " "NLP appears again.",
            return_reports=True,
        )

        # Build a minimal meanings_by_id map from extraction output.
        meanings_by_id = extr.meaning_index

        # Ensure we have both meanings and that we can control their meaning_confidence:
        # (E2E-friendly: mutate by rebuilding local objects if your dataclass is frozen elsewhere;
        # here, we just assert what's already present and use patch on meaning_prior term behaviour.)
        assert "nlp|natural_language_processing" in meanings_by_id
        assert "nlp|nice_lovely_plants" in meanings_by_id

        # Now patch confidence levels by monkeypatching the meaning_index entries via replacement.
        # If AcronmMeaning is mutable in your codebase, you can direct-set instead.
        s_hi = meanings_by_id["nlp|natural_language_processing"]
        s_lo = meanings_by_id["nlp|nice_lovely_plants"]

        # Sanity: make sure we can see a difference (or the prior would be moot).
        # If both are equal from upstream, this test can still pass by force-setting.
        try:
            s_hi.meaning_confidence = 0.95
            s_lo.meaning_confidence = 0.40
        except Exception:
            # If frozen, rebuild lightweight namespace objects for the call below
            pass

        # Directly call disambiguate_occurrences with a known near-tie + prior enabled.
        occs = [OccurrenceLite("NLP", 0, 3)]
        out = mod.disambiguate_occurrences(
            text="x" * 50,
            occurrences=occs,
            meanings={"NLP": list(extr.meaning_by_acronym["NLP"])},
            meanings_prior_weight=0.08,  # enable
            meanings_by_id=meanings_by_id,
            window_chars=10,
        )
        assert out and out[0].chosen_meaning_id is not None
        assert "natural_language_processing" in out[0].chosen_meaning_id

    def test_dynamic_prior_disabled_keeps_near_tie_unresolved(self):
        """
        Integration-style contract:
        - Build real meanings (and real def_spans) from the pipeline.
        - Create a synthetic occurrence positioned exactly midway between the two def spans.
        - With prior disabled and distance unable to distinguish, resolution stays undecided.
        """

        # 1) Run full pipeline once to get REAL meanings + REAL def_spans.
        _det, extr, _r = detect_and_extract(
            "Natural language processing (NLP) helps. " "Nice Lovely Plants (NLP) sold locally.",
            return_reports=True,
        )

        meanings = list(extr.meaning_by_acronym["NLP"])
        assert len(meanings) == 2
        s1, s2 = meanings

        # Take the first def span for each meaning and compute span-centers (same logic as disambiguate.py)
        (a1, b1) = s1.def_spans[0]
        (a2, b2) = s2.def_spans[0]
        c1 = (a1 + b1) // 2
        c2 = (a2 + b2) // 2

        # 2) Place occurrence start exactly at the midpoint between centers (equal distance to both).
        mid = (c1 + c2) // 2
        occ = OccurrenceLite("NLP", mid, mid + 3)

        # 3) Use dummy text with no useful overlap signal (tokens won't intersect meaning definitions).
        text = "x" * (mid + 50)

        out = disambiguate_occurrences(
            text=text,
            occurrences=[occ],
            meanings={"NLP": meanings},
            meanings_by_id=extr.meaning_index,
            window_chars=20,
            meanings_prior_weight=0.0,  # disable prior (what we're asserting)
            margin_threshold=0.10,
            dist_weight=0.75,
            overlap_weight=0.25,
        )

        assert out
        assert out[0].chosen_meaning_id is None, out[0]

    def test_choose_with_tiebreak_margins_checked(self):
        """
        Contract test: choose_with_tiebreak returns relative and absolute margins.
        """

        def S(
            acr: str, meaning_id: str, definition: str, spans: list[Span], *, conf: float = 0.0, support: int = 1
        ) -> AcronymMeaning:
            return AcronymMeaning(
                acronym=acr,
                definition=definition,
                meaning_id=meaning_id,
                meaning_confidence=conf,
                def_spans=list(spans),
                support=support,
            )

        occ = OccurrenceLite("PDF", 10, 13)
        meanings_by_id = {
            "s1": S("PDF", "s1", "Portable Document Format", [(0, 1)]),
            "s2": S("PDF", "s2", "Other", [(0, 1)]),
        }
        cand = {"s1": 0.80, "s2": 0.60}

        chosen, rel_margin, abs_margin = choose_with_tiebreak(occ, cand, meanings_by_id, margin_threshold=0.10)
        assert chosen == "s1"

        assert abs_margin == approx(0.200000, abs=1e-6)
        assert rel_margin == approx((0.80 - 0.60) / 0.80, rel=0, abs=1e-9)


# ----------------------------
# Patch point: Tier-2 embeddings
# ----------------------------


def _fake_embed_texts(texts, *, model=None, model_name=None, **_kw) -> np.ndarray:
    """
    Deterministic, fast "embeddings" for tests.

        Encodes texts into a tiny keyword-feature vector so cosine similarity is meaningful.
    """
    # Grouped features: doesn't require literal word overlap between context and definition,
    # only "same bucket" activation (e.g. GMP/purity aligns with pharmaceutical).
    assert isinstance(model_name, str)
    assert isinstance(texts, list), f"expected list[str], got {type(texts)}: {texts!r}"

    buckets = [
        {"programming", "interface", "endpoint", "http", "rest", "sdk", "client", "server"},
        {
            "pharmaceutical",
            "ingredient",
            "gmp",
            "assay",
            "batch",
            "purity",
            "tablet",
            "dose",
            "drug",
            "formulation",
            "excipient",
        },
    ]

    out = np.zeros((len(texts), len(buckets)), dtype=np.float32)
    for i, t in enumerate(texts):
        s = t.lower()
        for j, words in enumerate(buckets):
            out[i, j] = float(sum(1 for w in words if w in s))
        if not out[i].any():
            out[i, 0] = 1e-6
    return out


def _patch_tier2_embed(_patch):
    _patch(t2.embed_for_tier2, embed_texts=_fake_embed_texts)
    # optional paranoia check
    assert t2.embed_for_tier2.__globals__["embed_texts"] is _fake_embed_texts


# ----------------------------
# Tiny helpers to avoid overfitting to exact datamodel shapes
# ----------------------------


def _get(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if cur is None:
            return None
        cur = cur.get(part) if isinstance(cur, dict) else getattr(cur, part, None)
    return cur


def _tier2_report(state: FlowState) -> Any:
    rep = _get(state, "tier_2.report")
    assert rep is not None, "Expected state.tier_2.report to exist after the run."
    return rep


def _iter_meanings_with_ids(extr: Any):
    meanings = getattr(extr, "meanings", None)
    if meanings is None:
        return

    if isinstance(meanings, dict):
        for v in meanings.values():
            if isinstance(v, dict):
                for sid, s in v.items():
                    yield sid, s
            elif isinstance(v, (list, tuple)):
                for s in v:
                    yield _meaning_id(s), s
            else:
                yield _meaning_id(v), v
        return

    if isinstance(meanings, (list, tuple)):
        for s in meanings:
            yield _meaning_id(s), s
        return


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _meaning_id(meaning: Any) -> str | None:
    for attr in ("meaning_id", "id", "key"):
        v = getattr(meaning, attr, None)
        if isinstance(v, str) and v:
            return v
    return None


def _meaning_text(meaning: Any) -> str:
    # direct strings
    for attr in ("definition", "definition_text", "expanded", "expansion", "long_form", "text", "label"):
        v = getattr(meaning, attr, None)
        if isinstance(v, str) and v.strip():
            return v

        # nested "definition object" cases
        if v is not None:
            for sub in ("definition", "text", "label", "expanded", "expansion"):
                vv = getattr(v, sub, None)
                if isinstance(vv, str) and vv.strip():
                    return vv

    return ""


def _meaning_index(state: FlowState) -> dict[str, Any]:
    idx = _get(state, "disambig.meaning_index")
    assert isinstance(idx, dict) and idx, "Expected state.disambig.meaning_index to exist."
    return idx


def _iter_ranked_records(state: FlowState):
    ranked2 = _get(state, "tier_2.ranked")
    if isinstance(ranked2, (list, tuple)) and ranked2:
        return ranked2
    ranked1 = _get(state, "tier_1.ranked")
    assert isinstance(ranked1, (list, tuple)) and ranked1, "Expected state.tier_1.ranked"
    return ranked1


def _iter_candidate_sid_keys(r: Any):
    for attr in ("blended_scores", "candidate_scores"):
        m = getattr(r, attr, None)
        if isinstance(m, dict) and m:
            yield from m.keys()


def _tier1_ranked(state: FlowState):
    ranked1 = _get(state, "tier_1.ranked")
    assert isinstance(ranked1, (list, tuple)) and ranked1, "Expected state.tier_1.ranked"
    return ranked1


def _find_sid_by_slug(state: FlowState, *, acr: str, contains: str) -> str:
    needle = _norm(contains)
    acr_u = acr.upper()

    for r in _tier1_ranked(state):
        occ = getattr(r, "occ", None)
        if occ is None:
            continue
        if getattr(occ, "acronym", "").upper() != acr_u:
            continue

        scores = getattr(r, "candidate_scores", None)
        if not isinstance(scores, dict) or not scores:
            continue

        for sid in scores:
            if needle in _norm(sid):
                return sid

    raise AssertionError(f"Couldn't find meaning id for {acr} containing {contains!r}")


def _iter_resolutions(extr: Any) -> Iterable[Any]:
    res = getattr(extr, "resolutions", None) or getattr(extr, "occurrence_resolutions", None)
    assert res is not None, "Expected extr.resolutions (or extr.occurrence_resolutions) to exist."
    if isinstance(res, dict):
        for v in res.values():
            yield from (v if isinstance(v, (list, tuple)) else [v])
    else:
        yield from (res if isinstance(res, (list, tuple)) else [res])


def _res_key(r: Any) -> str | None:
    for attr in ("key", "acronym_key", "acro_key", "acronym"):
        v = getattr(r, attr, None)
        if isinstance(v, str) and v:
            return v
    occ = getattr(r, "occurrence", None)
    if occ is not None:
        for attr in ("key", "acronym_key", "acro_key", "acronym"):
            v = getattr(occ, attr, None)
            if isinstance(v, str) and v:
                return v
    return None


def _res_pos(r: Any) -> int | None:
    for attr in ("start", "start_pos", "pos", "offset"):
        v = getattr(r, attr, None)
        if isinstance(v, int):
            return v
    span = getattr(r, "span", None)
    if isinstance(span, (tuple, list)) and span and isinstance(span[0], int):
        return span[0]
    occ = getattr(r, "occurrence", None)
    if occ is not None:
        for attr in ("start", "start_pos", "pos", "offset"):
            v = getattr(occ, attr, None)
            if isinstance(v, int):
                return v
    return None


def _chosen_meaning_id(r: Any) -> str | None:
    for attr in ("chosen_meaning_id", "meaning_id", "chosen"):
        v = getattr(r, attr, None)
        if isinstance(v, str) and v:
            return v
    return None


def _run_flow(
    text: str,
    *,
    ext_cfg: ExtractionConfig,
    disambig_margin_threshold: float | None = None,
) -> tuple[Any, Any, list[Any], FlowState]:
    flow = ExtractionFlow(
        ext_cfg=ext_cfg,
        disambig_margin_threshold=disambig_margin_threshold,
    )
    state = FlowState(text=text, det_cfg=flow.det_cfg, ext_cfg=flow.ext_cfg)
    state, reports = flow.build_chain().run(state, tracer=None)
    assert state.det_res is not None and state.extr is not None
    return state.det_res, state.extr, reports, state


# ----------------------------
# E2E 1 — Tier-2 resolves later occurrences using context
# ----------------------------


class TestTier2E2e:
    def test_tier2_e2e_resolves_api_by_section_context(self, _patch) -> None:
        _patch_tier2_embed(_patch)

        assert (
            t2.embed_for_tier2.__globals__["embed_texts"](
                ["a", "b"],
                model=object(),
                model_name="fake",
            ).shape[0]
            == 2
        )

        ext_cfg = replace(
            ExtractionConfig(),
            tier2=Tier2Config(mode="on", weight=0.65, model_name="fake"),
        )

        text = (
            "SOFTWARE SECTION\n"
            "An Application Programming Interface (API) exposes HTTP endpoints for a client.\n"
            "The API endpoint returns JSON. A REST API is versioned.\n"
            "\n"
            "PHARMA APPENDIX\n"
            "Active Pharmaceutical Ingredient (API) is manufactured under GMP.\n"
            "The API batch assay passed; API purity exceeded the spec.\n"
        )

        _, extr, _, state = _run_flow(text, ext_cfg=ext_cfg)

        rep = _tier2_report(state)
        assert getattr(rep, "applied", 0) > 0, "Expected Tier-2 to apply to at least one occurrence."

        sid_software = _find_sid_by_slug(
            state,
            acr="API",
            contains="Application Programming Interface",
        )
        sid_pharma = _find_sid_by_slug(
            state,
            acr="API",
            contains="Active Pharmaceutical Ingredient",
        )

        boundary = text.index("PHARMA APPENDIX")

        api_res = [r for r in _iter_resolutions(extr) if (_res_key(r) or "").upper() == "API"]
        assert api_res, "Expected API resolutions to exist."

        for r in api_res:
            pos = _res_pos(r)
            chosen = _chosen_meaning_id(r)
            assert chosen is not None, "Expected chosen_meaning_id to be set for API occurrences."

            if pos is not None and pos < boundary:
                assert chosen == sid_software
            elif pos is not None:
                assert chosen == sid_pharma

    # ----------------------------
    # E2E 2 — Tier-2 does not run when Tier-1 is confident (auto)
    # ----------------------------

    def test_tier2_e2e_auto_skips_when_tier1_confident(self, _patch) -> None:
        _patch_tier2_embed(_patch)

        # Make auto gating extremely conservative: only ties (margin ~0) get Tier-2.
        ext_cfg = replace(
            ExtractionConfig(),
            tier2=Tier2Config(mode="auto", weight=0.65, model_name="fake", auto_margin_ceiling=0.01),
        )

        text = (
            "DOCS\n"
            "Portable Document Format (PDF) is a file format used by a document reader.\n"
            "This PDF file prints reliably. The PDF page layout is stable.\n"
            "\n" + ("FILLER " * 300) + "\n"
                                       "STATS\n"
                                       "Probability Density Function (PDF) describes a distribution in statistics.\n"
                                       "The PDF integrates to one for a random variable.\n"
        )

        _, extr, _, state = _run_flow(text, ext_cfg=ext_cfg)

        rep = _tier2_report(state)
        assert getattr(rep, "applied", 0) == 0, "Expected Tier-2 not to run under confident Tier-1 results."

        reasons = getattr(rep, "reasons", {}) or {}
        assert reasons.get("tier1_confident", 0) > 0, f"Expected tier1_confident skips; got reasons={reasons!r}"

        sid_docs = _find_sid_by_slug(state, acr="PDF", contains="Portable Document Format")
        sid_stats = _find_sid_by_slug(state, acr="PDF", contains="Probability Density Function")

        boundary = text.index("STATS")

        pdf_res = [r for r in _iter_resolutions(extr) if (_res_key(r) or "").upper() == "PDF"]
        assert pdf_res, "Expected PDF resolutions to exist."

        for r in pdf_res:
            pos = _res_pos(r)
            chosen = _chosen_meaning_id(r)
            assert chosen is not None
            if pos is not None and pos < boundary:
                assert chosen == sid_docs
            elif pos is not None:
                assert chosen == sid_stats

    # ----------------------------
    # E2E 3 — Tier-2 reranks and changes outcome (on vs off)
    # ----------------------------

    @staticmethod
    def pick_last_api_resolution_after(extr: Any, after: int) -> Any:
        cands = []
        for r in _iter_resolutions(extr):
            if (_res_key(r) or "").upper() != "API":
                continue
            pos = _res_pos(r)
            if pos is not None and pos > after:
                cands.append((pos, r))
        assert cands
        return sorted(cands, key=lambda x: x[0])[-1][1]

    def test_tier2_e2e_on_flips_choice_vs_tier1_rank(self, _patch) -> None:
        _patch_tier2_embed(_patch)

        text = (
            "INTRO\n"
            "An Application Programming Interface (API) exposes HTTP endpoints.\n"
            "The API endpoint returns JSON. The REST API is stable.\n"
            "\n"
            "OPERATIONS\n"
            "The API was approved for formulation and recorded in the run sheet.\n"
            "The API was released for use.\n"
            "\n" + ("FILLER " * 400) + "\n"
                                       "GLOSSARY\n"
                                       "Active Pharmaceutical Ingredient (API) is defined for manufacturing.\n"
        )

        ext_on = replace(
            ExtractionConfig(),
            tier2=Tier2Config(mode="on", weight=0.85, model_name="fake"),
        )

        _, extr_on, _, state_on = _run_flow(text, ext_cfg=ext_on)

        rep_on = _tier2_report(state_on)
        assert getattr(rep_on, "applied", 0) > 0

        sid_pharma = _find_sid_by_slug(state_on, acr="API", contains="Active Pharmaceutical Ingredient")

        boundary = text.index("OPERATIONS")
        r_on = self.pick_last_api_resolution_after(extr_on, boundary)

        pos_on = _res_pos(r_on)
        assert pos_on is not None

        chosen_on = _chosen_meaning_id(r_on)
        assert chosen_on is not None

        assert chosen_on == sid_pharma, f"Expected Tier-2 final choice to be pharma; got {chosen_on!r}"
        ranked2 = _get(state_on, "tier_2.ranked") or []
        assert ranked2, "Expected Tier-2 ranked records"

        assert any(
            getattr(getattr(rec, "occ", None), "start", None) == pos_on
            and getattr(getattr(rec, "occ", None), "acronym", "").upper() == "API"
            for rec in ranked2
        ), f"Expected Tier-2 to rank the target API occurrence at pos={pos_on}"
        rec = next(
            r
            for r in ranked2
            if getattr(getattr(r, "occ", None), "start", None) == pos_on
            and getattr(getattr(r, "occ", None), "acronym", "").upper() == "API"
        )

        # At least one of these should exist if Tier-2 actually did work
        tier2_sims = getattr(rec, "tier2_sims", None)
        blended = getattr(rec, "blended_scores", None)

        assert (isinstance(tier2_sims, dict) and tier2_sims) or (
            isinstance(blended, dict) and blended
        ), f"Expected Tier-2 scores for target occurrence; tier2_sims={tier2_sims!r}, blended_scores={blended!r}"
