from plainera_unacronym.nlp import Detector
from plainera_unacronym.nlp.common.types import ExtractionResult, OccurrenceLite, _compute_strategy
from plainera_unacronym.nlp.detection.cleanup.post import post_detect_cleanup
from plainera_unacronym.nlp.extraction.anchored.extract import extract_near_firsts
from plainera_unacronym.nlp.extraction.backref.extract import extract_sentence_backrefs
from plainera_unacronym.nlp.extraction.core.defs import dedupe_defs, defs_from_picks
from plainera_unacronym.nlp.extraction.senses.disambiguate import disambiguate_occurrences
from plainera_unacronym.nlp.extraction.senses.sense_build import build_senses
from plainera_unacronym.nlp.extraction.strategies.gapfill import fill_missing_from_defs
from plainera_unacronym.nlp.extraction.strategies.harvest import extract_defs_all_occurrences

from .stages import StageResult
from .state import FlowState


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


def st_gapfill(s: FlowState) -> StageResult[FlowState]:
    """Fill missing picks using definitions extracted by other strategies.

    For acronym keys where `s.picks[key]` is None, selects the best matching
    definition from `s.all_defs` (typically based on proximity/confidence via
    `fill_missing_from_defs`) and fills `s.picks`. Updates coverage metrics and
    `s.missing_keys`, and records a summary in `s.last_info`.

    Args:
        s (FlowState): Mutable flow state. Must already contain `s.det_res` and `s.all_defs`.

    Returns:
        StageResult[FlowState]: Updated flow state plus a human-readable note.

    Raises:
        AssertionError: If `s.det_res` is None (detect stage not run).
    """
    assert s.det_res is not None
    missing = [k for k, v in s.picks.items() if v is None]

    filled_any = False
    if missing:
        fills = fill_missing_from_defs(s.text, firsts=s.det_res.unique_acronyms, det_cfg=s.det_cfg, defs=s.all_defs)
        for k in missing:
            if s.picks[k] is None:
                picked = fills.get(k)
                if picked is not None:
                    s.picks[k] = picked
                    filled_any = True
    used = []
    if any(s.anchored_defs):
        used.append("anchored")
    if any(s.harvested_defs):
        used.append("harvest")
    if any(s.backref_defs):
        used.append("backref")
    if filled_any:
        used.append("gapfill")

    has_gapfill = filled_any
    has_anchored = bool(s.anchored_defs)
    has_harvest = bool(s.harvested_defs)
    has_global = False  # set this based on your pipeline
    # NOTE: backref is not represented in your type, so don't put it into the strategy string.

    s.strategy = _compute_strategy(
        has_gapfill=has_gapfill,
        has_global=has_global,
        has_anchored=has_anchored,
        has_harvest=has_harvest,
    )

    s.coverage = (len(s.picks) - sum(1 for v in s.picks.values() if v is None)) / max(1, len(s.picks))
    s.missing_keys = tuple(sorted(k for k, v in s.picks.items() if v is None))
    s.last_info = f"{s.strategy} coverage={s.coverage:.2%} missing={len(s.missing_keys)}"
    return StageResult(s, s.last_info)


def st_senses_and_assemble(
    s: FlowState, *, disambig_window_chars: int, disambig_margin_threshold: float
) -> StageResult[FlowState]:
    """Build senses, disambiguate occurrences, and assemble the final ExtractionResult.

    Builds sense candidates from `s.all_defs`, then performs occurrence-level
    disambiguation over the document to choose a sense per occurrence. Populates
    `s.extr` with picks, definitions, senses, and disambiguation outputs.
    Records counts (total senses and undecided occurrences) in `s.last_info`.

    Args:
        s (FlowState): Mutable flow state. Must already contain `s.det_res` and `s.all_defs`.
        disambig_window_chars (int): Context window size (chars) for disambiguation.
        disambig_margin_threshold (float): Minimum score margin required to auto-select a sense.

    Returns:
        StageResult[FlowState]: Updated flow state plus a human-readable note.

    Raises:
        AssertionError: If `s.det_res` is None (detect stage not run).
    """
    assert s.det_res is not None

    senses_by_acr = build_senses(s.all_defs)
    sense_index = {x.sense_id: x for xs in senses_by_acr.values() for x in xs}
    occs = [OccurrenceLite(o.acronym, o.start_offset, o.end_offset) for o in s.det_res.occurrences]

    resolutions = disambiguate_occurrences(
        text=s.text,
        occurrences=occs,
        senses=senses_by_acr,
        window_chars=disambig_window_chars,
        margin_threshold=disambig_margin_threshold,
    )
    undecided = [r for r in resolutions if r.chosen_sense_id is None]
    ambiguous = tuple(sorted(k for k, v in senses_by_acr.items() if len(v) > 1))

    s.extr = ExtractionResult(
        picks=s.picks,
        definitions=s.all_defs,
        extraction_strategy=s.strategy,
        coverage=s.coverage,
        missing_keys=s.missing_keys,
        senses_by_acronym=senses_by_acr,
        sense_index=sense_index,
        resolutions=resolutions,
        ambiguous_keys=ambiguous,
        undecided=undecided,
    )
    s.last_info = f"senses={sum(len(v) for v in senses_by_acr.values())}, undecided={len(undecided)}"
    return StageResult(s, s.last_info)
