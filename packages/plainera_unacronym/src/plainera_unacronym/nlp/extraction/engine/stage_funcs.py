from collections import Counter

from plainera_unacronym.nlp import Detector
from plainera_unacronym.nlp.common.types import ExtractionResult, OccurrenceLite, OccurrenceResolution
from plainera_unacronym.nlp.detection.cleanup.post import post_detect_cleanup
from plainera_unacronym.nlp.extraction.anchored.extract import extract_near_firsts
from plainera_unacronym.nlp.extraction.backref.extract import extract_sentence_backrefs
from plainera_unacronym.nlp.extraction.core.defs import dedupe_defs, defs_from_picks
from plainera_unacronym.nlp.extraction.engine.stages import StageResult
from plainera_unacronym.nlp.extraction.engine.state import FlowState
from plainera_unacronym.nlp.extraction.senses.disambiguate import disambiguate_occurrences, choose_with_tiebreak, \
    NEAR_TIE_GAP
from plainera_unacronym.nlp.extraction.senses.sense_build import build_senses
from plainera_unacronym.nlp.extraction.strategies.harvest import extract_defs_all_occurrences
from plainera_unacronym.nlp.extraction.strategies.pick_resolution import (
    backfill_missing_picks_from_defs,
    build_defs_index,
    patch_pick_provenance,
)
from plainera_unacronym.nlp.extraction.tiers.tier_2 import collect_tier2_inputs, apply_tier2_reranks, _embed_for_tier2
from plainera_unacronym.nlp.extraction.tiers.types import Tier2OccurrenceRanking, Tier2Report, Tier2SkipReason, \
    Tier1OccurrenceRanking


def st_detect(s: FlowState) -> StageResult[FlowState]:
    """Run detection to find acronym occurrences and first occurrences.

    Populates `s.det_res` with the detector output and records a short summary
    into `s.last_info`.

    Args:
        s (FlowState): Mutable flow state containing input text and detector config.

    Returns:
        StageResult[FlowState]: Updated flow state plus a human-readable note.
    """
    det = Detector(config=s.det_cfg).detect(s.text)
    s.det_res = det
    s.last_info = f"firsts={len(det.unique_acronyms)} occs={len(det.occurrences)}"
    return StageResult(s, s.last_info)


def st_post_detect_cleanup(s: FlowState) -> StageResult[FlowState]:
    """Apply post-detection cleanup to remove/adjust invalid occurrences.

    Runs `post_detect_cleanup` on the detector result, updates `s.det_res` with
    the cleaned result, stores dropped occurrences in `s.cleanup_dropped`, and
    records the cleanup summary in `s.last_info`.

    Args:
        s (FlowState): Mutable flow state. Must already contain `s.det_res`.

    Returns:
        StageResult[FlowState]: Updated flow state plus a human-readable note.

    Raises:
        AssertionError: If `s.det_res` is None (detect stage not run).
    """
    assert s.det_res is not None
    cleaned, summary, dropped = post_detect_cleanup(s.text, s.det_res, s.det_cfg)
    s.det_res = cleaned
    s.cleanup_dropped = dropped
    s.last_info = summary
    return StageResult(s, s.last_info)


def st_picks_first_occurrence_anchored(s: FlowState, *, window_left: int, window_right: int) -> StageResult[FlowState]:
    """Extract near first occurrences using anchored patterns within a local window.

    Uses `extract_near_firsts` around each first occurrence (FO) to produce
    `InTextPick` candidates. Stores results in `s.picks` and records pick
    coverage in `s.last_info`.

    Args:
        s (FlowState): Mutable flow state. Must already contain `s.det_res`.
        window_left (int): Characters to include to the left of each FO.
        window_right (int): Characters to include to the right of each FO.

    Returns:
        StageResult[FlowState]: Updated flow state plus a human-readable note.

    Raises:
        AssertionError: If `s.det_res` is None (detect stage not run).
    """
    assert s.det_res is not None
    s.picks = extract_near_firsts(
        s.text,
        firsts=s.det_res.unique_acronyms,
        cfg=s.ext_cfg,
        window_left=window_left,
        window_right=window_right,
    )
    got = sum(1 for v in s.picks.values() if v)
    s.last_info = f"anchored picks {got}/{len(s.picks)}"
    return StageResult(s, s.last_info)


def st_defs_from_first_occurrence_picks(s: FlowState) -> StageResult[FlowState]:
    """Convert anchored picks into concrete extracted definition records.

    Converts `s.picks` to a list of `ExtractedDefinition` objects and stores them
    in `s.anchored_defs`. Updates `s.last_info` with a count summary.

    Args:
        s (FlowState): Mutable flow state containing `s.picks`.

    Returns:
        StageResult[FlowState]: Updated flow state plus a human-readable note.
    """
    s.anchored_defs = defs_from_picks(s.text, s.picks)
    s.last_info = f"anchored defs={len(s.anchored_defs)}"
    return StageResult(s, s.last_info)


def st_defs_scan_all_occurrences(s: FlowState) -> StageResult[FlowState]:
    """Harvest additional definitions across all occurrences.

    Runs the harvest strategy across all detected occurrences (not only first
    occurrences). Stores results in `s.harvested_defs` and records a count in
    `s.last_info`.

    Args:
        s (FlowState): Mutable flow state. Must already contain `s.det_res`.

    Returns:
        StageResult[FlowState]: Updated flow state plus a human-readable note.

    Raises:
        AssertionError: If `s.det_res` is None (detect stage not run).
    """
    assert s.det_res is not None
    s.harvested_defs = extract_defs_all_occurrences(s.text, s.det_res.occurrences, s.ext_cfg)
    s.last_info = f"harvested={len(s.harvested_defs)}"
    return StageResult(s, s.last_info)


def st_sentence_backref(s: FlowState) -> StageResult[FlowState]:
    """Extract sentence back-references where a definition precedes an acronym.

    Runs the back-reference strategy to find definitions in prior sentences for
    acronyms that appear later without an inline/parenthetical long-form.
    Stores results in `s.backref_defs` and records a count in `s.last_info`.

    Args:
        s (FlowState): Mutable flow state. Must already contain `s.det_res`.

    Returns:
        StageResult[FlowState]: Updated flow state plus a human-readable note.

    Raises:
        AssertionError: If `s.det_res` is None (detect stage not run).
    """
    assert s.det_res is not None
    s.backref_defs = extract_sentence_backrefs(text=s.text, firsts=s.det_res.unique_acronyms, cfg=s.ext_cfg)
    s.last_info = f"backref={len(s.backref_defs)}"
    return StageResult(s, s.last_info)


def st_merge(s: FlowState) -> StageResult[FlowState]:
    """Merge and deduplicate all extracted definitions from all strategies.

    Concatenates definitions from anchored, harvested, global, and backref
    sources, then removes duplicates using `dedupe_defs`. Stores results in
    `s.all_defs` and records the unique count in `s.last_info`.

    Args:
        s (FlowState): Mutable flow state containing per-strategy definition lists.

    Returns:
        StageResult[FlowState]: Updated flow state plus a human-readable note.
    """
    s.all_defs = dedupe_defs(s.anchored_defs + s.harvested_defs + s.global_defs + s.backref_defs)
    s.last_info = f"merged unique={len(s.all_defs)}"
    return StageResult(s, s.last_info)


def st_finalise_picks(s: FlowState) -> StageResult[FlowState]:
    """
    Finalise `s.picks` by filling any missing acronyms from `s.all_defs`.

    Keeps any existing picks intact, only backfilling `None` entries. Populates
    pick provenance (`kind`, `route`, `reasons`) from the winning definition.

    Args:
        s: FlowState for the pipeline stage. Requires `s.det_res` to be present.

    Returns:
        StageResult containing the mutated FlowState and a short info string.
    """
    assert s.det_res is not None

    defs_index = build_defs_index(
        s.all_defs,
        allow_chars=s.det_cfg.allow_chars,
        dotted_mode=s.det_cfg.dotted_display,
    )

    backfill_missing_picks_from_defs(
        s.picks,
        defs_index=defs_index,
        unique_acronyms=s.det_res.unique_acronyms,
    )

    patch_pick_provenance(s.picks, default_route="first_occurrence_anchored")

    s.coverage = (len(s.picks) - sum(1 for v in s.picks.values() if v is None)) / max(1, len(s.picks))
    s.missing_keys = tuple(sorted(k for k, v in s.picks.items() if v is None))
    s.last_info = f"coverage={s.coverage:.2%} missing={len(s.missing_keys)}"
    return StageResult(s, s.last_info)

def st_tier1_build_senses(s: FlowState) -> StageResult[FlowState]:
    """
        Tier-1 setup: build senses and lightweight occurrences for disambiguation.

        Constructs the Tier-1 disambiguation working set from the current extraction
        state:

        - Derives `s.disambig.tier1.senses_by_acronym` from `s.all_defs` via
          `build_senses()`.
        - Builds `s.disambig.tier1.sense_index` for O(1) lookup by `sense_id`.
        - Projects detector occurrences (`s.det_res.occurrences`) into a minimal
          `OccurrenceLite` list (`acronym`, `start_offset`, `end_offset`) suitable
          for scoring and reranking stages.

        This stage does not score or choose senses; it only prepares data structures
        consumed by subsequent Tier-1/Tier-2 stages.

        Args:
            s: FlowState for the pipeline stage. Requires `s.det_res` and `s.all_defs`
                to be populated.

        Returns:
            StageResult containing the mutated FlowState and a short info string
            summarising the number of senses and occurrences prepared.
    """
    assert s.det_res is not None

    t1 = s.disambig.tier1
    t1.senses_by_acronym = build_senses(s.all_defs)
    t1.sense_index = {x.sense_id: x for xs in t1.senses_by_acronym.values() for x in xs}
    t1.occurrences = [OccurrenceLite(o.acronym, o.start_offset, o.end_offset) for o in s.det_res.occurrences]

    s.last_info = f"senses={sum(len(v) for v in t1.senses_by_acronym.values())} occs={len(t1.occurrences)}"
    return StageResult(s, s.last_info)



def st_tier1_score_occurrences(
    s: FlowState, *, window_chars: int, margin_threshold: float
) -> StageResult[FlowState]:
    """
    Tier-1: score each occurrence against candidate senses and produce a provisional choice.

    Runs the heuristic disambiguation pass over the prepared Tier-1 working set
    (`s.disambig.tier1.occurrences` and `s.disambig.tier1.senses_by_acronym`),
    computing per-occurrence candidate scores and (optionally) selecting a
    provisional winning sense.

    The underlying scorer (`disambiguate_occurrences`) is expected to return, per
    occurrence:
      - a stable mapping of candidate `sense_id -> score` (`candidate_scores`)
      - a provisional winner (`chosen_sense_id`) or `None` if undecided
      - tie diagnostics (`gap`, `margin`) used by later selection/assembly stages

    Results are normalised into `Tier1OccurrenceRanking` entries and stored in
    `s.disambig.tier1.ranked`. This stage does not mutate extracted definitions;
    it only annotates occurrences with ranking metadata.

    Args:
        s: FlowState for the pipeline stage. Requires `s.det_res` and Tier-1
            working data (senses/occurrences) to be prepared (typically by
            `st_tier1_build_senses`).
        window_chars: Number of characters to include on each side of the
            occurrence when deriving the scoring context window.
        margin_threshold: Minimum relative margin required for the heuristic
            scorer to consider an occurrence “decided”; otherwise the winner is
            left as `None`.

    Returns:
        StageResult containing the mutated FlowState and a short info string
        reporting the number of ranked occurrences and how many remain undecided.
    """
    assert s.det_res is not None

    t1 = s.disambig.tier1

    res = disambiguate_occurrences(
        text=s.text,
        occurrences=t1.occurrences,
        senses=t1.senses_by_acronym,
        window_chars=window_chars,
        margin_threshold=margin_threshold,
        senses_by_id=t1.sense_index,
    )

    t1.ranked = [
        Tier1OccurrenceRanking(
            occ=OccurrenceLite(r.acronym, r.start, r.end),
            candidate_scores=r.candidate_scores,
            chosen_sense_id=r.chosen_sense_id,
            gap=r.gap,
            margin=r.margin,
        )
        for r in res
    ]

    undec = sum(1 for r in t1.ranked if r.chosen_sense_id is None)
    s.last_info = f"ranked={len(t1.ranked)} undecided={undec}"
    return StageResult(s, s.last_info)


def st_tier2_semantic_rerank(s: FlowState, *, window_chars: int) -> StageResult[FlowState]:
    """
    Tier-2: optional semantic rerank of Tier-1 candidates (no acceptance changes).

    Applies embedding-based similarity scoring to reorder/adjust Tier-1 candidate
    scores for eligible occurrences. Tier-2 is conservative: it only runs when
    enabled and when Tier-1 did not already decide for an occurrence.

    Args:
        s: FlowState for the pipeline stage. Requires `s.det_res` and Tier-1
            rankings in `s.disambig.tier1.ranked`.
        window_chars: Context window size around each occurrence.

    Returns:
        StageResult containing the mutated FlowState and a short info string.
    """
    assert s.det_res is not None

    t1 = s.disambig.tier1
    t2 = s.disambig.tier2

    tier2_cfg = getattr(s.ext_cfg, "tier2", None)
    enabled = bool(getattr(tier2_cfg, "enabled", False))

    reasons: Counter[Tier2SkipReason] = Counter()

    if not enabled:
        n = len(t1.ranked)
        t2.ranked = []
        t2.report = Tier2Report(applied=0, skipped=n, reasons={"disabled": n})
        s.last_info = "tier2=skipped(disabled)"
        return StageResult(s, s.last_info)

    weight = float(getattr(tier2_cfg, "weight", 0.35))
    model_name = str(getattr(tier2_cfg, "model_name", "all-MiniLM-L6-v2"))

    ranked2, eligible = collect_tier2_inputs(
        text=s.text,
        t1_ranked=t1.ranked,
        sense_index=t1.sense_index,
        window_chars=window_chars,
        reasons=reasons,
    )

    if not eligible:
        t2.ranked = []
        t2.report = Tier2Report(applied=0, skipped=len(t1.ranked), reasons=dict(reasons))
        s.last_info = "tier2=skipped(nothing_eligible)"
        return StageResult(s, s.last_info)

    batch = _embed_for_tier2(model_name, eligible)
    if batch is None:
        reasons["model_unavailable"] += len(eligible)
        t2.ranked = ranked2
        t2.report = Tier2Report(applied=0, skipped=len(ranked2), reasons=dict(reasons))
        s.last_info = "tier2=skipped(model_unavailable)"
        return StageResult(s, s.last_info)

    applied = apply_tier2_reranks(ranked2=ranked2, eligible=eligible, batch=batch, weight=weight)

    t2.ranked = ranked2
    t2.report = Tier2Report(applied=applied, skipped=len(ranked2) - applied, reasons=dict(reasons))
    s.last_info = f"tier2=applied({applied}) skipped({len(ranked2)-applied})"
    return StageResult(s, s.last_info)


def st_tiers_select_and_assemble(
    s: FlowState, *, margin_threshold: float
) -> StageResult[FlowState]:
    """
    Final selection + result assembly using Tier-1 rankings (optionally Tier-2 reordered scores).

    Converts disambiguation work products into the public `ExtractionResult`.

    Selection policy:

    - Tier-1 produces a provisional winner (`chosen_sense_id`) and diagnostics
      (`gap`, `margin`) per occurrence.
    - Tier-2, if applied for a given occurrence, may provide `blended_scores`
      that *reorder/adjust* candidate scores.
    - The acceptance policy (margin/tie-break rules) is applied once at this
      stage:
        - If Tier-2 blended scores exist for an occurrence, selection is computed
          from those scores via `choose_with_tiebreak`.
        - Otherwise the Tier-1 provisional decision is preserved verbatim (no
          recomputation).

    The stage then assembles:

    - `resolutions`: per-occurrence `OccurrenceResolution` entries
    - `ambiguous_keys`: acronyms with multiple candidate senses
    - `undecided`: occurrences with no chosen sense

    and writes the final `ExtractionResult` to `s.extr`.

    Args:
        s: FlowState for the pipeline stage. Requires `s.det_res` to be present
            and Tier-1 rankings to have been computed (typically by
            `st_tier1_score_occurrences`). Tier-2 rankings may be present but are
            optional.
        margin_threshold: Minimum relative margin required to accept a winner
            when computing selection from Tier-2 blended scores.

    Returns:
        StageResult containing the mutated FlowState (with `s.extr` populated)
        and a short info string summarising senses and the number of undecided
        occurrences.
    """
    assert s.det_res is not None

    t1 = s.disambig.tier1
    t2 = s.disambig.tier2

    tier2_by_key: dict[tuple[str, int, int], Tier2OccurrenceRanking] = {
        (r2.occ.acronym, r2.occ.start, r2.occ.end): r2
        for r2 in (t2.ranked or [])
    }

    resolutions: list[OccurrenceResolution] = []

    for r1 in t1.ranked:
        key = (r1.occ.acronym, r1.occ.start, r1.occ.end)
        r2 = tier2_by_key.get(key)

        if r2 and r2.applied and r2.blended_scores:
            chosen, rel_margin, gap = choose_with_tiebreak(
                r1.occ,
                r2.blended_scores,
                t1.sense_index,
                margin_threshold=margin_threshold,
                near_tie_margin=NEAR_TIE_GAP,
            )
            cand_scores = r2.blended_scores
        else:
            chosen, rel_margin, gap = r1.chosen_sense_id, r1.margin, r1.gap
            cand_scores = r1.candidate_scores

        resolutions.append(
            OccurrenceResolution(
                r1.occ.acronym,
                r1.occ.start,
                r1.occ.end,
                chosen,
                cand_scores,
                gap=gap,
                margin=rel_margin,
            )
        )

    undecided = [r for r in resolutions if r.chosen_sense_id is None]
    ambiguous = tuple(sorted(k for k, v in t1.senses_by_acronym.items() if len(v) > 1))

    s.extr = ExtractionResult(
        picks=s.picks,
        definitions=s.all_defs,
        coverage=s.coverage,
        missing_keys=s.missing_keys,
        senses_by_acronym=t1.senses_by_acronym,
        sense_index=t1.sense_index,
        resolutions=resolutions,
        ambiguous_keys=ambiguous,
        undecided=undecided,
    )

    s.last_info = f"senses={sum(len(v) for v in t1.senses_by_acronym.values())}, undecided={len(undecided)}"
    return StageResult(s, s.last_info)
