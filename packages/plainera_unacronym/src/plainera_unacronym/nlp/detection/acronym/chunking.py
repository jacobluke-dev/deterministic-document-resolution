from observability.logger.levels import LogLevel
from observability.logger.message_logger import message_logger


from plainera_unacronym.nlp.common.types import AcronymDetectorConfig, OccurrenceBuildError, Occurrence
from plainera_unacronym.nlp.detection.acronym.builders import build_occurrence_from_match
from plainera_unacronym.nlp.detection.heuristics.context import blacklist_context_drop
from plainera_unacronym.nlp.detection.heuristics.core import calc_score, threshold_len
from plainera_unacronym.wiring.observability import sink


def score_chunk_worker(cfg: AcronymDetectorConfig, text: str, cands: list[tuple[str, int, int]]) -> list[Occurrence]:
    """
    Score and filter a chunk of candidate acronym spans.

    Processes (surface, start, end) tuples, drops blacklisted contexts and
    below-threshold matches, and builds `Occurrence` objects for accepted items.
    Intended for use in a `ProcessPoolExecutor`.

    Args:
        cfg: Detection configuration used for scoring and thresholds.
        text: Source text the candidate spans refer to.
        cands: Candidate tuples as (surface, start_offset, end_offset).

    Returns:
        list[Occurrence]: Accepted occurrences for this chunk.

    """
    out: list[Occurrence] = []
    for surface, s, e in cands:
        if blacklist_context_drop(surface, text, s, e, cfg):
            continue
        conf = calc_score(surface, text, s, e, cfg)
        eff = threshold_len(surface, cfg.allow_chars)
        th = cfg.min_confidence_by_len.get(eff, cfg.min_confidence_default)
        if conf < th:
            continue
        try:
            occ, _ = build_occurrence_from_match(cfg, text, surface, s, e, conf)
        except OccurrenceBuildError as err:
            if getattr(cfg, "debug_anomalies", False):
                message_logger(
                    "detector.bad_occurrence",
                    level=LogLevel.DEBUG,
                    logger_type="nlp",
                    details={"reason": str(err), "surface": surface, "s": s, "e": e},
                    db_sink=sink,
                )
            continue
        out.append(occ)
    return out
