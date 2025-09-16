from concurrent.futures import ProcessPoolExecutor
import asyncio
from typing import Optional
from dataclasses import replace as dc_replace

from observability.logger.decorator import logger
from observability.logger.levels import LogLevel
from observability.logger.message_logger import message_logger
from plainera_unacronym.nlp.config import ALLOW_CHARS, DOT_MODE
from plainera_unacronym.nlp.heuristics.core import score, threshold_len, normalize_key, context_window, compile_pattern, \
    iter_candidates_with, reason_tags
from plainera_unacronym.nlp.heuristics.general import blacklist_context_drop, strip_terminal_plural
from plainera_unacronym.nlp.nlp_helpers import top_n_values, _cfg_fingerprint
from plainera_unacronym.nlp.plugins.activation import autodetect_domains
from plainera_unacronym.nlp.types import DetectorConfig, DetectorResult, Occurrence, FirstOccurrence
from plainera_unacronym.wiring.composition import sink

DEFAULT_CONFIG = DetectorConfig()
ALLOW_CHARS_DEFAULTS = ALLOW_CHARS
DEFAULT_DOT_MODE = DOT_MODE


def _build_occurrence_from_match(
    cfg: DetectorConfig,
    text: str,
    surface: str,
    s: int,
    e: int,
    conf: float,
) -> tuple[Occurrence, str]:
    """
    Build a single Occurrence with consistent policy:
    - dotted display policy ('strip'|'preserve')
    - optional trailing '.' preservation (without touching the regex)
    - normalized_key for display
    - (optional) canonical_key if you want for dedupe (here we return only display key;
      use normalize_key(..., dotted_mode='strip') separately if you dedupe canonically)
    Returns (occurrence, display_key_for_firsts).
    """
    display_mode = getattr(cfg, "dotted_display", "strip")          # 'strip' or 'preserve'
    has_trailing_dot = (e < len(text) and text[e] == ".")

    # Surface to display (may include the trailing dot if preserving)
    surface_for_display = surface + "." if (display_mode == "preserve" and has_trailing_dot) else surface
    end_for_occ = e + 1 if (display_mode == "preserve" and has_trailing_dot) else e

    base = strip_terminal_plural(surface_for_display)

    display_key = normalize_key(
        base,
        cfg.allow_chars,
        dotted_mode=display_mode,
    )

    ctx = context_window(text, s, end_for_occ, cfg.window_chars)

    # Optional reason tags; keep the same in serial + parallel
    rsn = tuple(reason_tags(surface, text, s, end_for_occ, cfg)) if getattr(cfg, "debug_reasons", False) else None

    # IMPORTANT: construct with keyword args to avoid field-order bugs
    occ = Occurrence(
        acronym=base,               # string; never assign an int here
        start_offset=s,
        end_offset=end_for_occ,
        confidence=conf,
        context_window=ctx,
        normalized_key=display_key,
        reasons=rsn,
    )
    return occ, display_key


def _score_chunk_worker(cfg: DetectorConfig, text: str, cands: list[tuple[str,int,int]]):
    out: list[Occurrence] = []
    for surface, s, e in cands:
        if blacklist_context_drop(surface, text, s, e, cfg):
            continue
        conf = score(surface, text, s, e, cfg)
        eff  = threshold_len(surface, cfg.allow_chars)
        th   = cfg.min_confidence_by_len.get(eff, cfg.min_confidence_default)
        if conf < th:
            continue
        occ, _ = _build_occurrence_from_match(cfg, text, surface, s, e, conf)
        out.append(occ)
    return out


class Detector:
    def __init__(self,
                 config: DetectorConfig = DEFAULT_CONFIG,
                 max_workers: Optional[int] = None):
        self.cfg = config
        self._pat = compile_pattern(config)  # precompiled once
        self._pool: Optional[ProcessPoolExecutor] = None
        self._max_workers = max_workers
        self.sink = sink

    def _with_auto_domains(self, text: str) -> DetectorConfig:
        auto = autodetect_domains(text, self.cfg)  # frozenset[str]
        if auto:
            merged = self.cfg.enabled_domains | auto
            if merged != self.cfg.enabled_domains:
                return dc_replace(self.cfg, enabled_domains=merged)
        return self.cfg

    @logger(message="detector.detect", db_sink="sink")  # decorator logs duration, function, args map
    def detect(self, text: str) -> DetectorResult:
        cfg0 = self.cfg
        cfg = self._with_auto_domains(text)

        if cfg is not cfg0:
            # log domains auto-added (diff only)
            added = sorted((cfg.enabled_domains or set()) - (cfg0.enabled_domains or set()))
            if added:
                message_logger(
                    "detector.autodetect_domains",
                    level=LogLevel.INFO,
                    logger_type="nlp",
                    args={"text_len": len(text)},
                    details={"added": added, "total_domains": len(cfg.enabled_domains or ())},
                    db_sink=self.sink,
                )

        total = dropped_blacklist = below_threshold = accepted = 0
        occurrences: list[Occurrence] = []
        firsts: dict[str, FirstOccurrence] = {}

        #  start event
        message_logger(
            "detector.detect.start",
            level=LogLevel.INFO,
            logger_type="message_logger.nlp",
            args={
                "text_len": len(text),
                "cfg_fp": _cfg_fingerprint(cfg),
                "window_chars": cfg.window_chars,
                "dotted_display": getattr(cfg, "dotted_display", "strip"),
            },
            db_sink=self.sink,
        )

        for surface, s, e in iter_candidates_with(text, cfg, self._pat):
            total += 1
            if blacklist_context_drop(surface, text, s, e, cfg):
                dropped_blacklist += 1
                continue

            conf = score(surface, text, s, e, cfg)
            eff = threshold_len(surface, cfg.allow_chars)
            th = cfg.min_confidence_by_len.get(eff, cfg.min_confidence_default)
            if conf < th:
                below_threshold += 1
                continue

            occ, display_key = _build_occurrence_from_match(cfg, text, surface, s, e, conf)
            occurrences.append(occ)
            accepted += 1

            if display_key not in firsts:
                firsts[display_key] = FirstOccurrence(
                    acronym=occ.acronym,
                    start_offset=occ.start_offset,
                    end_offset=occ.end_offset,
                    confidence=conf,
                    normalized_key=display_key,
                )

        message_logger(
            "detector.detect.summary",
            level=LogLevel.INFO,
            logger_type="message_logger.nlp",
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

        return DetectorResult(unique_acronyms=firsts, occurrences=occurrences)

    @logger(message="detector.parallel", db_sink="sink")
    def detect_parallel(self, text: str, threshold: int = 1000, chunk_size: int = 256) -> DetectorResult:
        cfg = self._with_auto_domains(text)
        cands = list(iter_candidates_with(text, cfg, self._pat))
        if len(cands) < threshold:
            return self.detect(text)

        if self._pool is None:
            from os import cpu_count
            self._pool = ProcessPoolExecutor(max_workers=self._max_workers)
            message_logger(
                "detector.pool.created",
                logger_type="message_logger.nlp",
                args={"max_workers": self._max_workers or cpu_count() or 1},
                db_sink=self.sink,
            )

        num_chunks = (len(cands) + chunk_size - 1) // chunk_size
        message_logger(
            "detector.parallel.selected",
            logger_type="message_logger.nlp",
            args={"candidates": len(cands), "chunk_size": chunk_size, "num_chunks": num_chunks},
            db_sink=self.sink,
        )

        futures = []
        for i in range(0, len(cands), chunk_size):
            futures.append(self._pool.submit(_score_chunk_worker, cfg, text, cands[i:i + chunk_size]))

        occurrences: list[Occurrence] = []
        for idx, f in enumerate(futures):
            try:
                occurrences.extend(f.result())
            except Exception as e:
                import traceback
                message_logger(
                    "detector.chunk.failed",
                    level=LogLevel.ERROR,
                    logger_type="message_logger.nlp",
                    args={"chunk_index": idx},
                    details={"error": str(e), "trace": traceback.format_exc()},
                    db_sink=self.sink,
                )

        firsts: dict[str, FirstOccurrence] = {}
        for occ in occurrences:
            if not isinstance(occ.acronym, str):
                message_logger(
                    "detector.bad_occurrence",
                    level=LogLevel.ERROR,
                    logger_type="message_logger.nlp",
                    details={"type": str(type(occ.acronym)), "occ": repr(occ)[:200]},
                    db_sink=self.sink,
                )
                continue

            display_key = getattr(occ, "normalized_key", normalize_key(
                occ.acronym, cfg.allow_chars, dotted_mode=getattr(cfg, "dotted_display", "strip")
            ))
            if display_key not in firsts:
                firsts[display_key] = FirstOccurrence(
                    acronym=occ.acronym,
                    start_offset=occ.start_offset,
                    end_offset=occ.end_offset,
                    confidence=occ.confidence,
                    normalized_key=display_key,
                )

        return DetectorResult(unique_acronyms=firsts, occurrences=occurrences)

    async def detect_async(self, text: str) -> DetectorResult:
        """
           Asynchronously run acronym detection without blocking the event loop.

           Async wrapper that offloads the sync detection pipeline to the event loop's default
           executor (a thread pool) by calling `detect_parallel` in a background thread. Use
           in FastAPI so the loop remains responsive while CPU-bound work executes off-loop.
           Depending on input size, the underlying `detect_parallel` method may further split work across processes.

           Args:
               text: The raw input text to scan for acronyms.
           Returns:
               DetectorResult: The detection result containing all occurrences and
               a mapping of first occurrences keyed by normalized acronym.
           Raises:
               Exception: Propagates any exception raised by the detection pipeline.
           See Also:
               detect: Synchronous, single-process path for smaller inputs.
               detect_parallel: Potentially multi-process path used under the hood for large inputs.
           """
        # Run parallel path off the event loop; won’t block FastAPI
        return await asyncio.to_thread(self.detect_parallel, text)
