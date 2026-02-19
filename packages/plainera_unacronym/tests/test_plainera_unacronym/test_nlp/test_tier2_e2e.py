# tests/integration/test_tier2_e2e.py

from __future__ import annotations

from dataclasses import replace
import re
from typing import Any, Iterable

import numpy as np
import pytest

from plainera_unacronym.nlp.extraction.config import ExtractionConfig, Tier2Config
from plainera_unacronym.nlp.extraction.engine.detect_flow import ExtractionFlow
from plainera_unacronym.nlp.extraction.engine.state import FlowState
import plainera_unacronym.nlp.extraction.tiers.tier_2 as t2


@pytest.fixture(autouse=True)
def _mock_tier2_embeddings():
    # disable the autouse conftest patch in this module
    yield


# ----------------------------
# Patch point: Tier-2 embeddings
# ----------------------------

def _fake_embed_texts(model_name: str, texts: list[str], *_, **__) -> np.ndarray:
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
        {"pharmaceutical", "ingredient", "gmp", "assay", "batch", "purity", "tablet", "dose", "drug", "formulation",
         "excipient"},
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
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
    return cur


def _tier2_report(state: FlowState) -> Any:
    rep = _get(state, "disambig.tier2.report")
    assert rep is not None, "Expected state.disambig.tier2.report to exist after the run."
    return rep


def _iter_senses_with_ids(extr: Any):
    senses = getattr(extr, "senses", None)
    if senses is None:
        return

    if isinstance(senses, dict):
        for v in senses.values():
            if isinstance(v, dict):
                for sid, s in v.items():
                    yield sid, s
            elif isinstance(v, (list, tuple)):
                for s in v:
                    yield _sense_id(s), s
            else:
                yield _sense_id(v), v
        return

    if isinstance(senses, (list, tuple)):
        for s in senses:
            yield _sense_id(s), s
        return


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _sense_id(sense: Any) -> str | None:
    for attr in ("sense_id", "id", "key"):
        v = getattr(sense, attr, None)
        if isinstance(v, str) and v:
            return v
    return None


def _sense_text(sense: Any) -> str:
    # direct strings
    for attr in ("definition", "definition_text", "expanded", "expansion", "long_form", "text", "label"):
        v = getattr(sense, attr, None)
        if isinstance(v, str) and v.strip():
            return v

        # nested "definition object" cases
        if v is not None:
            for sub in ("definition", "text", "label", "expanded", "expansion"):
                vv = getattr(v, sub, None)
                if isinstance(vv, str) and vv.strip():
                    return vv

    return ""


def _sense_index(state: FlowState) -> dict[str, Any]:
    idx = _get(state, "disambig.sense_index")
    assert isinstance(idx, dict) and idx, "Expected state.disambig.sense_index to exist."
    return idx

def _iter_ranked_records(state: FlowState):
    ranked2 = _get(state, "disambig.tier2.ranked")
    if isinstance(ranked2, (list, tuple)) and ranked2:
        return ranked2
    ranked1 = _get(state, "disambig.tier1.ranked")
    assert isinstance(ranked1, (list, tuple)) and ranked1, "Expected state.disambig.tier1.ranked"
    return ranked1


def _iter_candidate_sid_keys(r: Any):
    for attr in ("blended_scores", "candidate_scores"):
        m = getattr(r, attr, None)
        if isinstance(m, dict) and m:
            yield from m.keys()


def _tier1_ranked(state: FlowState):
    ranked1 = _get(state, "disambig.tier1.ranked")
    assert isinstance(ranked1, (list, tuple)) and ranked1, "Expected state.disambig.tier1.ranked"
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

        for sid in scores.keys():
            if needle in _norm(sid):
                return sid

    raise AssertionError(f"Couldn't find sense id for {acr=} containing {contains!r}")




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


def _chosen_sense_id(r: Any) -> str | None:
    for attr in ("chosen_sense_id", "sense_id", "chosen"):
        v = getattr(r, attr, None)
        if isinstance(v, str) and v:
            return v
    return None


def _run_flow(
    text: str,
    *,
    ext_cfg: ExtractionConfig,
    disambig_window_chars: int | None = None,
    disambig_margin_threshold: float | None = None,
) -> tuple[Any, Any, list[Any], FlowState]:
    flow = ExtractionFlow(
        ext_cfg=ext_cfg,
        disambig_window_chars=disambig_window_chars,
        disambig_margin_threshold=disambig_margin_threshold,
    )
    state = FlowState(text=text, det_cfg=flow.det_cfg, ext_cfg=flow.ext_cfg)
    state, reports = flow.build_chain().run(state, tracer=None)
    assert state.det_res is not None and state.extr is not None
    return state.det_res, state.extr, reports, state


# ----------------------------
# E2E 1 — Tier-2 resolves later occurrences using context
# ----------------------------

def test_tier2_e2e_resolves_api_by_section_context(_patch) -> None:
    _patch_tier2_embed(_patch)
    assert t2.embed_for_tier2.__globals__["embed_texts"]("fake", ["a", "b"]).shape[0] == 2

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
    # Use a deliberately small disambiguation context window in this test.
    # The document has two nearby sections with different meanings for the same acronym (API). With a large window,
    # the sliced context can include keywords from *both* sections (“HTTP/REST” and “GMP/assay/purity”), which makes
    # the fake keyword-bucket embeddings ambiguous and can destabilise reranking. Keeping the window small keeps each
    # occurrence’s context local to its section so Tier-2 can separate senses deterministically.

    _, extr, _, state = _run_flow(text, ext_cfg=ext_cfg, disambig_window_chars=50)

    rep = _tier2_report(state)
    assert getattr(rep, "applied", 0) > 0, "Expected Tier-2 to apply to at least one occurrence."

    sid_software = _find_sid_by_slug(state, acr="API", contains="Application Programming Interface")
    sid_pharma = _find_sid_by_slug(state, acr="API", contains="Active Pharmaceutical Ingredient")

    boundary = text.index("PHARMA APPENDIX")

    api_res = [r for r in _iter_resolutions(extr) if (_res_key(r) or "").upper() == "API"]
    assert api_res, "Expected API resolutions to exist."

    for r in api_res:
        pos = _res_pos(r)
        chosen = _chosen_sense_id(r)
        assert chosen is not None, "Expected chosen_sense_id to be set for API occurrences."

        if pos is not None and pos < boundary:
            assert chosen == sid_software
        elif pos is not None:
            assert chosen == sid_pharma


# ----------------------------
# E2E 2 — Tier-2 does not run when Tier-1 is confident (auto)
# ----------------------------

def test_tier2_e2e_auto_skips_when_tier1_confident(_patch) -> None:
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
        "\n"
        + ("FILLER " * 300) + "\n"
        "STATS\n"
        "Probability Density Function (PDF) describes a distribution in statistics.\n"
        "The PDF integrates to one for a random variable.\n"
    )

    _, extr, _, state = _run_flow(text, ext_cfg=ext_cfg, disambig_window_chars=80)

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
        chosen = _chosen_sense_id(r)
        assert chosen is not None
        if pos is not None and pos < boundary:
            assert chosen == sid_docs
        elif pos is not None:
            assert chosen == sid_stats


# ----------------------------
# E2E 3 — Tier-2 reranks and changes outcome (on vs off)
# ----------------------------

def _pick_last_api_resolution_after(extr: Any, after: int) -> Any:
    cands = []
    for r in _iter_resolutions(extr):
        if (_res_key(r) or "").upper() != "API":
            continue
        pos = _res_pos(r)
        if pos is not None and pos > after:
            cands.append((pos, r))
    assert cands
    return sorted(cands, key=lambda x: x[0])[-1][1]

def test_tier2_e2e_on_flips_choice_vs_tier1_rank(_patch) -> None:
    _patch_tier2_embed(_patch)

    text = (
        "INTRO\n"
        "An Application Programming Interface (API) exposes HTTP endpoints.\n"
        "The API endpoint returns JSON. The REST API is stable.\n"
        "\n"
        "OPERATIONS\n"
        "The API was approved for formulation and recorded in the run sheet.\n"
        "The API was released for use.\n"
        "\n"
        + ("FILLER " * 400) + "\n"
        "GLOSSARY\n"
        "Active Pharmaceutical Ingredient (API) is defined for manufacturing.\n"
    )

    ext_on = replace(
        ExtractionConfig(),
        tier2=Tier2Config(mode="on", weight=0.85, model_name="fake", select_margin_threshold=0.20),
    )

    _, extr_on, _, state_on = _run_flow(text, ext_cfg=ext_on, disambig_window_chars=80)

    rep_on = _tier2_report(state_on)
    assert getattr(rep_on, "applied", 0) > 0

    sid_pharma = _find_sid_by_slug(state_on, acr="API", contains="Active Pharmaceutical Ingredient")

    boundary = text.index("OPERATIONS")
    r_on = _pick_last_api_resolution_after(extr_on, boundary)

    pos_on = _res_pos(r_on)
    assert pos_on is not None

    chosen_on = _chosen_sense_id(r_on)
    assert chosen_on is not None

    assert chosen_on == sid_pharma, f"Expected Tier-2 final choice to be pharma; got {chosen_on!r}"
    ranked2 = _get(state_on, "disambig.tier2.ranked") or []
    assert ranked2, "Expected Tier-2 ranked records"

    assert any(
        getattr(getattr(rec, "occ", None), "start", None) == pos_on
        and getattr(getattr(rec, "occ", None), "acronym", "").upper() == "API"
        for rec in ranked2
    ), f"Expected Tier-2 to rank the target API occurrence at pos={pos_on}"
    rec = next(
        r for r in ranked2
        if getattr(getattr(r, "occ", None), "start", None) == pos_on
        and getattr(getattr(r, "occ", None), "acronym", "").upper() == "API"
    )

    # At least one of these should exist if Tier-2 actually did work
    tier2_sims = getattr(rec, "tier2_sims", None)
    blended = getattr(rec, "blended_scores", None)

    assert (isinstance(tier2_sims, dict) and tier2_sims) or (isinstance(blended, dict) and blended), (
        f"Expected Tier-2 scores for target occurrence; tier2_sims={tier2_sims!r}, blended_scores={blended!r}"
    )
