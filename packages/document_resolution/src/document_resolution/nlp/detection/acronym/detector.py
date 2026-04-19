from dataclasses import replace as dc_replace

from observability.logger.decorator import logger
from observability.logger.levels import LogLevel
from observability.logger.message_logger import message_logger

from document_resolution.nlp.common.shared import normalize_acronym_key
from document_resolution.nlp.common.types import (
    AcronymDetectorConfig,
    AcronymDetectorResult,
    FirstOccurrence,
    Occurrence,
    OccurrenceBuildError,
)
from document_resolution.nlp.detection.base import BaseDetector
from document_resolution.nlp.detection.heuristics.context import blacklist_context_drop
from document_resolution.nlp.detection.heuristics.core import (
    boost_confidence_if_whitelisted,
    calc_score,
    iter_acronym_candidates,
    threshold_len,
)
from document_resolution.nlp.detection.heuristics.inline_cues import boost_confidence_if_inline_cue
from document_resolution.nlp.detection.nlp_helpers import cfg_fingerprint, top_n_values
from document_resolution.nlp.plugins.activation import autodetect_domains

from .builders import build_occurrence_from_match
from .chunking import score_chunk_worker
from .compiler import compile_acronym_pattern

DEFAULT_CONFIG = AcronymDetectorConfig()


class AcronymDetector(BaseDetector[AcronymDetectorResult]):
    def __init__(
        self,
        config: AcronymDetectorConfig = DEFAULT_CONFIG,
        max_workers: int | None = None,
        sink=None,
    ):
        super().__init__(config=config, max_workers=max_workers, sink=sink)
        self._pat = compile_acronym_pattern(config)

    def _with_auto_domains(self, text: str) -> AcronymDetectorConfig:
        """Return a config updated with any auto-detected domains.

        Args:
            text: Input text to scan for domain cues.

        Returns:
            Config with augmented `enabled_domains` when new domains are detected.
        """
        auto = autodetect_domains(text, self.cfg)
        if auto:
            merged = self.cfg.enabled_domains | auto
            if merged != self.cfg.enabled_domains:
                return dc_replace(self.cfg, enabled_domains=merged)
        return self.cfg

    @logger(message="acronym_detector.detect", db_sink="sink")
    def detect(self, text: str) -> AcronymDetectorResult:
        """Run acronym detection over the input text.

        Applies auto-domain detection, scores and filters candidate spans, and returns
        both accepted occurrences and first occurrences keyed by normalised acronym.

        Args:
            text: Input text to analyse.

        Returns:
            Detection result containing `occurrences` and `unique_acronyms`.
        """
        cfg0 = self.cfg
        cfg = self._with_auto_domains(text)

        if cfg is not cfg0:
            added = sorted(cfg.enabled_domains - cfg0.enabled_domains)
            if added:
                message_logger(
                    "acronym_detector.autodetect_domains",
                    level=LogLevel.INFO,
                    logger_type="nlp",
                    args={"text_len": len(text)},
                    details={"added": added, "total_domains": len(cfg.enabled_domains)},
                    db_sink=self.sink,
                )

        total = dropped_blacklist = below_threshold = accepted = 0
        occurrences: list[Occurrence] = []
        firsts: dict[str, FirstOccurrence] = {}

        message_logger(
            "acronym_detector.detect.start",
            level=LogLevel.INFO,
            logger_type="nlp",
            args={
                "text_len": len(text),
                "cfg_fp": cfg_fingerprint(cfg),
            },
            db_sink=self.sink,
        )

        for surface, s, e in iter_acronym_candidates(text, cfg, self._pat):
            total += 1

            if blacklist_context_drop(surface, text, s, e, cfg):
                dropped_blacklist += 1
                continue

            # 1) raw score
            conf = calc_score(surface, text, s, e, cfg)

            # 2) compute length bucket / threshold (using our existing helper)
            eff = threshold_len(surface, cfg.allow_chars)
            th = cfg.min_confidence_by_len.get(eff, cfg.min_confidence_default)

            # 3) optional boost for allow-listed 2-letter acronyms (boost the SCORE, not the threshold)
            conf = boost_confidence_if_whitelisted(surface, conf, cfg)  # returns min(conf+boost, 0.99)
            conf = boost_confidence_if_inline_cue(surface, text, e, conf)

            if conf < th:
                below_threshold += 1
                continue

            try:
                occ, display_key = build_occurrence_from_match(cfg, text, surface, s, e, conf)
            except OccurrenceBuildError as err:
                message_logger(
                    "acronym_detector.bad_occurrence",
                    level=LogLevel.ERROR,
                    logger_type="nlp",
                    details={"reason": str(err), "surface": surface, "s": s, "e": e},
                    db_sink=self.sink,
                )
                continue
            occurrences.append(occ)
            accepted += 1

            if display_key not in firsts:
                firsts[display_key] = FirstOccurrence(
                    acronym=occ.acronym,
                    start_offset=occ.start_offset,
                    end_offset=occ.end_offset,
                    occurrence_confidence=conf,
                    normalized_key=display_key,
                )

        message_logger(
            "acronym_detector.detect.summary",
            level=LogLevel.INFO,
            logger_type="nlp",
            args={
                "text_len": len(text),
                "candidates": total,
                "dropped_blacklist": dropped_blacklist,
                "below_threshold": below_threshold,
                "accepted": accepted,
                "unique": len(firsts),
            },
            details={"top": top_n_values(firsts, 5)},
            db_sink=self.sink,
        )

        return AcronymDetectorResult(unique_acronyms=firsts, occurrences=occurrences)

    @logger(message="acronym_detector.parallel", db_sink="sink")
    def detect_parallel(self, text: str, threshold: int = 1000, chunk_size: int = 256) -> AcronymDetectorResult:
        """Run acronym detection with optional parallel scoring for large inputs.

        Falls back to `detect()` when the candidate count is below `threshold`.
        Otherwise, scores candidates in chunks and rebuilds first occurrences from the
        merged results.

        Args:
            text: Input text to analyse.
            threshold: Minimum candidate count required for parallel execution.
            chunk_size: Number of candidates per worker task.

        Returns:
            Detection result containing `occurrences` and `unique_acronyms`.
        """
        cfg = self._with_auto_domains(text)
        cands = list(iter_acronym_candidates(text, cfg, self._pat))

        if len(cands) < threshold:
            return self.detect(text)

        pool = self._get_or_create_pool()
        futures = [
            pool.submit(score_chunk_worker, cfg, text, cands[i : i + chunk_size])
            for i in range(0, len(cands), chunk_size)
        ]

        occurrences: list[Occurrence] = []
        failed_chunks = 0
        for idx, future in enumerate(futures):
            try:
                occurrences.extend(future.result())
            except Exception as e:
                failed_chunks += 1
                import traceback

                message_logger(
                    "acronym_detector.chunk.failed",
                    level=LogLevel.ERROR,
                    logger_type="nlp",
                    args={"chunk_index": idx},
                    details={"error": str(e), "trace": traceback.format_exc()},
                    db_sink=self.sink,
                )

        if failed_chunks == len(futures):
            occurrences = score_chunk_worker(cfg, text, cands)

        firsts: dict[str, FirstOccurrence] = {}
        for occ in occurrences:
            display_key = getattr(occ, "normalized_key", None)

            if not isinstance(display_key, str) or not display_key:
                display_key = normalize_acronym_key(
                    occ.acronym,
                    cfg.allow_chars,
                    dotted_mode=cfg.dotted_display,
                )

            if display_key not in firsts:
                firsts[display_key] = FirstOccurrence(
                    acronym=occ.acronym,
                    start_offset=occ.start_offset,
                    end_offset=occ.end_offset,
                    occurrence_confidence=occ.occurrence_confidence,
                    normalized_key=display_key,
                )

        return AcronymDetectorResult(unique_acronyms=firsts, occurrences=occurrences)
