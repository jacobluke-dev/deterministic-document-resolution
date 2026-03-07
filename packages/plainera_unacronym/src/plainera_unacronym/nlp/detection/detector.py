import asyncio
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace as dc_replace
from typing import Optional

from observability.logger.decorator import logger
from observability.logger.levels import LogLevel
from observability.logger.message_logger import message_logger

from plainera_unacronym.nlp.common.constants_regex import ALLOW_CHARS
from plainera_unacronym.nlp.common.shared import normalize_acronym_key, strip_trailing_punct_str
from plainera_unacronym.nlp.common.types import (
    DetectorConfig,
    DetectorResult,
    FirstOccurrence,
    Occurrence,
    OccurrenceBuildError,
)
from plainera_unacronym.nlp.plugins.activation import autodetect_domains
from plainera_unacronym.wiring.composition import sink

from .heuristics.context import blacklist_context_drop
from .heuristics.core import (
    boost_confidence_if_whitelisted,
    calc_score,
    compile_pattern,
    context_window,
    iter_candidates_with,
    reason_tags,
    threshold_len,
)
from .heuristics.general import strip_terminal_plural
from .heuristics.inline_cues import boost_confidence_if_inline_cue
from .nlp_helpers import _cfg_fingerprint, top_n_values

DEFAULT_CONFIG = DetectorConfig()
ALLOW_CHARS_DEFAULTS = ALLOW_CHARS


def _adjust_end_for_trailing_dot(cfg: DetectorConfig, text: str, s: int, e: int) -> int:
    """
    Apply the dotted-display policy to an occurrence end-offset.

    If `cfg.dotted_display == "preserve"` and the character immediately following the
    matched span (`text[e]`) is a literal '.', advance the end offset by one so the
    occurrence span includes that trailing dot (e.g. matching "U.S" in "U.S." yields an
    end offset that includes the final period). In "strip" mode (or when no trailing dot
    exists), the end offset is returned unchanged.

    This function validates that the resulting span `[s, end_for_occ)` is a well-formed
    slice into `text`.

    Args:
        cfg: Detection configuration; reads `dotted_display` ("strip" or "preserve").
        text: The full source text the offsets refer to.
        s: Start offset (inclusive) of the matched surface.
        e: End offset (exclusive) of the matched surface (before trailing-dot adjustment).

    Returns:
        int: The adjusted end offset (exclusive) to use for the occurrence.

    Raises:
        OccurrenceBuildError: If the adjusted offsets are invalid (out of bounds or
            start/end ordering is wrong).
    """
    display_mode = getattr(cfg, "dotted_display", "strip")

    has_trailing_dot = e < len(text) and text[e] == "."

    # Surface to display (may include the trailing dot if preserving)
    end_for_occ = e + 1 if (display_mode == "preserve" and has_trailing_dot) else e
    if not (0 <= s < end_for_occ <= len(text)):
        raise OccurrenceBuildError("bad_offsets")
    return end_for_occ


def _normalize_surface_for_key(surface: str) -> str:
    """
    Normalise a matched surface into (base_surface, key_base) for occurrence/key construction.

    The `base_surface` is produced by stripping terminal plural suffixes from fully-uppercase
    acronym tokens (e.g. "GPUs" -> "GPU", "CPU's" -> "CPU"). The `key_base` is then derived
    from `base_surface` by removing trailing punctuation via `strip_trailing_punct_str()`,
    ensuring acronym/key strings do not end with punctuation.

    Note:
        This does not canonicalize internal punctuation (e.g. dotted initialisms) — that is
        handled later by `normalize_acronym_key(..., dotted_mode=cfg.dotted_display)`.

    Args:
        surface: Raw matched surface form, typically `text[s:e]`.

    Returns:
        str:
            - key_base: `base_surface` with trailing punctuation stripped, suitable for
              key normalization and for storing as the occurrence acronym.
    """

    # IMPORTANT: strip trailing punct from base so acronym/key never has terminal dot
    key_base = strip_trailing_punct_str(strip_terminal_plural(surface))
    return key_base


def _build_occurrence_from_match(
    cfg: DetectorConfig,
    text: str,
    surface: str,
    s: int,
    e: int,
    conf: float,
) -> tuple[Occurrence, str]:
    """
    Build an `Occurrence` from a single candidate span and return it with its display key.

    Applies the dotted-display policy from `cfg.dotted_display` ("strip" or "preserve").
    preserve mode extends the occurrence span to include a trailing dot (offset only);
    the acronym/key do not retain terminal punctuation
    When `cfg.debug_reasons` is enabled, reason tags are attached.

    Args:
        cfg: Detection configuration
        text: Full source text.
        surface: Matched surface form (typically `text[s:e]`).
        s: Start offset (inclusive) of the match.
        e: End offset (exclusive) of the match before any trailing-dot adjustment.
        conf: Confidence score for this match.

    Returns:
        tuple[Occurrence, str]: The constructed `Occurrence` and its normalized
        display key (used for deduping/first-occurrence tracking).

    Raises:
        OccurrenceBuildError: If occurrence is invalid, not of type `str`, or empty, or poor
        offsets.

    Notes:
        * Trailing-dot handling is offset-based only; no regex is modified.
        * `context_window` is derived from the adjusted `(s, end_for_occ)` span.
        * Normalization uses `normalize_key(..., dotted_mode=cfg.dotted_display)`.
    """
    end_for_occ = _adjust_end_for_trailing_dot(cfg, text, s, e)

    key_base = _normalize_surface_for_key(surface)

    if not key_base.strip():
        raise OccurrenceBuildError("empty_acronym")

    display_key = normalize_acronym_key(
        key_base,
        cfg.allow_chars,
        dotted_mode=cfg.dotted_display,  # this governs INTERNAL dot handling (U.S.A vs USA)
    )
    if not display_key:
        raise OccurrenceBuildError("empty_display_key")

    ctx = context_window(text, s, end_for_occ, cfg.window_chars)

    # Optional reason tags; keep the same in serial + parallel
    rsn = tuple(reason_tags(surface, text, s, end_for_occ, cfg)) if getattr(cfg, "debug_reasons", False) else None

    occ = Occurrence(
        acronym=key_base,
        start_offset=s,
        end_offset=end_for_occ,
        occurrence_confidence=conf,
        segment_window=ctx,
        normalized_key=display_key,
        reasons=rsn,
    )
    return occ, display_key


def _score_chunk_worker(cfg: DetectorConfig, text: str, cands: list[tuple[str, int, int]]) -> list[Occurrence]:
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
            occ, _ = _build_occurrence_from_match(cfg, text, surface, s, e, conf)
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


class Detector:
    def __init__(self, config: DetectorConfig = DEFAULT_CONFIG, max_workers: Optional[int] = None):
        self.cfg = config
        self._pat = compile_pattern(config)  # precompiled once
        self._pool: Optional[ProcessPoolExecutor] = None
        self._max_workers = max_workers
        self.sink = sink

    def _with_auto_domains(self, text: str) -> DetectorConfig:
        """
        Return a config updated with any domains auto-detected from the text.

        Merges inferred domains with the current `enabled_domains`. If nothing new
        is detected, the existing config is returned unchanged.

        Args:
            text: Input text to scan for domain cues.

        Returns:
            DetectorConfig | dict[str|Any]: Config with augmented `enabled_domains` when applicable.
        """
        auto = autodetect_domains(text, self.cfg)  # frozenset[str]
        if auto:
            merged = self.cfg.enabled_domains | auto
            if merged != self.cfg.enabled_domains:
                return dc_replace(self.cfg, enabled_domains=merged)
        return self.cfg

    @logger(message="detector.detect", db_sink="sink")  # decorator logs duration, function, args map
    def detect(self, text: str) -> DetectorResult:
        """
        Run acronym detection over the given text and return matches.

        Applies auto-domain detection to augment the active config, scans for
        candidate acronyms, filters by context/thresholds, and builds both the
        full list of occurrences and the first-occurrence map per normalized key.

        Structured logging:
          - Emits a lightweight “start” and “summary” event.
          - The @logger decorator also records duration/function metadata.
          - message_logs at points providing structured detail.

        Args:
            text: Input text to analyze.

        Returns:
            DetectorResult: Contains `occurrences` and `unique_acronyms`.
        """
        cfg0 = self.cfg
        cfg = self._with_auto_domains(text)

        if cfg is not cfg0:
            # log domains auto-added (diff only)
            added = sorted(cfg.enabled_domains - cfg0.enabled_domains)
            if added:
                message_logger(
                    "detector.autodetect_domains",
                    level=LogLevel.INFO,
                    logger_type="nlp",
                    args={"text_len": len(text)},
                    details={"added": added, "total_domains": len(cfg.enabled_domains)},
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

            # 1) raw score
            conf = calc_score(surface, text, s, e, cfg)

            # 2) compute length bucket / threshold (using our existing helper)
            eff = threshold_len(surface, cfg.allow_chars)
            th = cfg.min_confidence_by_len.get(eff, cfg.min_confidence_default)

            # 3) optional boost for allow-listed 2-letter acronyms (boost the SCORE, not the threshold)
            conf = boost_confidence_if_whitelisted(surface, conf, cfg)  # returns min(conf+boost, 0.99)
            conf = boost_confidence_if_inline_cue(surface, text, e, conf)

            # 4) gate
            if conf < th:
                below_threshold += 1
                continue

            try:
                occ, display_key = _build_occurrence_from_match(cfg, text, surface, s, e, conf)
            except OccurrenceBuildError as err:
                message_logger(
                    "detector.bad_occurrence",
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
            "detector.detect.summary",
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

        return DetectorResult(unique_acronyms=firsts, occurrences=occurrences)

    @logger(message="detector.parallel", db_sink="sink")
    def detect_parallel(self, text: str, threshold: int = 1000, chunk_size: int = 256) -> DetectorResult:
        """
        Run detection with optional multiprocess fan-out for large inputs.

        Computes candidates, and if their count is below `threshold` defers to
        `detect()`. Otherwise, splits work into `chunk_size` batches and processes
        them via a lazily-created `ProcessPoolExecutor`, then merges results and
        builds the first-occurrence map.

        Structured logging:
          - Emits pool creation and parallel-selection events.
          - Logs per-chunk failures (with a traceback).
          - The @logger decorator also records duration/function metadata.
          - message_logs at points providing structured detail.

        Args:
            text: Input text to analyze.
            threshold: Minimum candidate count to trigger parallel execution.
            chunk_size: Number of candidates per process task.

        Returns:
            DetectorResult: Contains `occurrences` and `unique_acronyms`.
        """
        cfg = self._with_auto_domains(text)
        cands = list(iter_candidates_with(text, cfg, self._pat))
        if len(cands) < threshold:
            return self.detect(text=text)

        if self._pool is None:
            from os import cpu_count

            self._pool = ProcessPoolExecutor(max_workers=self._max_workers)
            message_logger(
                "detector.pool.created",
                logger_type="nlp",
                args={"max_workers": self._max_workers or cpu_count() or 1},
                db_sink=self.sink,
            )

        num_chunks = (len(cands) + chunk_size - 1) // chunk_size
        message_logger(
            "detector.parallel.selected",
            logger_type="nlp",
            args={"candidates": len(cands), "chunk_size": chunk_size, "num_chunks": num_chunks},
            db_sink=self.sink,
        )

        futures = []
        for i in range(0, len(cands), chunk_size):
            futures.append(self._pool.submit(_score_chunk_worker, cfg, text, cands[i : i + chunk_size]))

        occurrences: list[Occurrence] = []
        for idx, f in enumerate(futures):
            try:
                occurrences.extend(f.result())
            except Exception as e:
                import traceback

                message_logger(
                    "detector.chunk.failed",
                    level=LogLevel.ERROR,
                    logger_type="nlp",
                    args={"chunk_index": idx},
                    details={"error": str(e), "trace": traceback.format_exc()},
                    db_sink=self.sink,
                )

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

        return DetectorResult(unique_acronyms=firsts, occurrences=occurrences)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown(wait=False, cancel_futures=True)
        return False

    def shutdown(self, *, wait: bool = False, cancel_futures: bool = True) -> None:
        """
        Shut down the internal ProcessPoolExecutor (if created).

        Call this from your application shutdown hook to avoid leaking processes.
        """
        pool = self._pool
        if pool is None:
            return

        pool.shutdown(wait=wait, cancel_futures=cancel_futures)
        self._pool = None

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
