from concurrent.futures import ProcessPoolExecutor
import asyncio
from typing import Optional

from plainera_unacronym.nlp.config import ALLOW_CHARS
from plainera_unacronym.nlp.heuristics.core import score, threshold_len, normalize_key, context_window, compile_pattern, \
    iter_candidates_with, iter_candidates, reason_tags
from plainera_unacronym.nlp.heuristics.general import blacklist_context_drop, strip_terminal_plural
from plainera_unacronym.nlp.types import DetectorConfig, DetectorResult, Occurrence, FirstOccurrence

DEFAULT_CONFIG = DetectorConfig()
ALLOW_CHARS_DEFAULTS = ALLOW_CHARS


def _score_chunk_worker(cfg: DetectorConfig, text: str, window_chars: int, cands):
    out: list[Occurrence] = []
    for surface, s, e in cands:
        if blacklist_context_drop(surface, text, s, e, cfg):
            continue
        conf = score(surface, text, s, e, cfg)
        eff = threshold_len(surface, cfg.allow_chars)
        th = cfg.min_confidence_by_len.get(eff, cfg.min_confidence_default)
        if conf < th:
            continue
        key = normalize_key(surface, cfg.allow_chars, cfg.enable_dotted)
        ctx = context_window(text, s, e, window_chars)
        rsn = tuple(reason_tags(surface, text, s, e, cfg)) if cfg.debug_reasons else None
        out.append(Occurrence(surface, s, e, conf, ctx, key, rsn))
    return out


class Detector:
    def __init__(self, config: DetectorConfig = DEFAULT_CONFIG, max_workers: Optional[int] = None):
        self.cfg = config
        self._pat = compile_pattern(config)  # precompiled once
        self._pool: Optional[ProcessPoolExecutor] = None
        self._max_workers = max_workers

    def detect(self, text: str) -> DetectorResult:
        occurrences: list[Occurrence] = []
        firsts: dict[str, FirstOccurrence] = {}

        for surface, s, e in iter_candidates_with(text, self.cfg, self._pat):
            if blacklist_context_drop(surface, text, s, e, self.cfg):
                continue
            conf = score(surface, text, s, e, self.cfg)
            eff = threshold_len(surface, self.cfg.allow_chars)
            th = self.cfg.min_confidence_by_len.get(eff, self.cfg.min_confidence_default)
            if conf < th:
                continue
            base = strip_terminal_plural(surface)
            key = normalize_key(
                base,
                self.cfg.allow_chars,
                self.cfg.enable_dotted,  # needed for "U.S." → "US"
            )
            ctx = context_window(text, s, e, self.cfg.window_chars)
            rsn = tuple(reason_tags(surface, text, s, e, self.cfg)) if self.cfg.debug_reasons else None

            occ = Occurrence(acronym=surface,
                             start_offset=s,
                             end_offset=e,
                             confidence=conf,
                             context_window=ctx,
                             normalized_key=key,
                             reasons=rsn)
            occurrences.append(occ)

            if key not in firsts:
                firsts[key] = FirstOccurrence(acronym=surface,
                                              start_offset=s,
                                              end_offset=e,
                                              confidence=conf,
                                              normalized_key=key)
        return DetectorResult(unique_acronyms=firsts, occurrences=occurrences)

    def detect_parallel(self, text: str, threshold: int = 1000, chunk_size: int = 256) -> DetectorResult:
        # Heuristic: small inputs are faster serially
        cands = list(iter_candidates_with(text, self.cfg, self._pat))
        if len(cands) < threshold:
            return self.detect(text)

        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self._max_workers)

        futures = []
        for i in range(0, len(cands), chunk_size):
            chunk = cands[i:i + chunk_size]
            futures.append(self._pool.submit(
                _score_chunk_worker, self.cfg, text, self.cfg.window_chars, chunk
            ))

        occurrences: list[Occurrence] = []
        for f in futures:
            occurrences.extend(f.result())

        # Build first-occurrence map
        firsts: dict[str, FirstOccurrence] = {}
        for occ in occurrences:
            key = normalize_key(occ.acronym, self.cfg.allow_chars, self.cfg.enable_dotted)
            if key not in firsts:
                firsts[key] = FirstOccurrence(occ.acronym, occ.start_offset, occ.end_offset, occ.confidence)
        return DetectorResult(unique_acronyms=firsts, occurrences=occurrences)

    async def detect_async(self, text: str) -> DetectorResult:
        loop = asyncio.get_running_loop()
        # Run parallel path off the event loop; won’t block FastAPI
        return await loop.run_in_executor(None, self.detect_parallel, text)


def detect_acronyms(text: str,
                    config: DetectorConfig = DEFAULT_CONFIG,
                    allowed_chars=ALLOW_CHARS_DEFAULTS) -> DetectorResult:
    """
    One-pass detector. Returns stable schema + first-occurrence map with normalized keys.
    """
    occurrences: list[Occurrence] = []
    firsts: dict[str, FirstOccurrence] = {}

    for surface, s, e in iter_candidates(text, config):
        if blacklist_context_drop(surface, text, s, e, config):
            continue

        conf = score(surface, text, s, e, config)
        ctx = context_window(text, s, e, config.window_chars)

        occ = Occurrence(
            acronym=surface,
            start_offset=s,
            end_offset=e,
            confidence=conf,
            context_window=ctx,
        )
        occurrences.append(occ)

        key = normalize_key(surface, allowed_chars, config.enable_dotted)
        if key not in firsts:
            firsts[key] = FirstOccurrence(
                acronym=surface,
                start_offset=s,
                end_offset=e,
                confidence=conf,
            )

    return DetectorResult(unique_acronyms=firsts, occurrences=occurrences)
