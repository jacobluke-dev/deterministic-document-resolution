from concurrent.futures import ProcessPoolExecutor
import asyncio
from typing import Optional

from src.plainera_unacronym.nlp.heuristics import (
    context_window, score, normalize_key, blacklist_context_drop,
    iter_candidates, compile_pattern, iter_candidates_with,
)
from src.plainera_unacronym.nlp.types import DetectorConfig, DetectorResult, Occurrence, FirstOccurrence

DEFAULT_CONFIG = DetectorConfig()


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
            ctx = context_window(text, s, e, self.cfg.window_chars)
            occ = Occurrence(acronym=surface, start_offset=s, end_offset=e, confidence=conf, context_window=ctx)
            occurrences.append(occ)
            key = normalize_key(surface)
            if key not in firsts:
                firsts[key] = FirstOccurrence(acronym=surface, start_offset=s, end_offset=e, confidence=conf)
        return DetectorResult(unique_acronyms=firsts, occurrences=occurrences)

    def detect_parallel(self, text: str, threshold: int = 1000, chunk_size: int = 256) -> DetectorResult:
        # Heuristic: small inputs are faster serially
        cands = list(iter_candidates_with(text, self.cfg, self._pat))
        if len(cands) < threshold:
            # Re-run serial path to keep code simple (or reuse scored objects if you prefer)
            return self.detect(text)

        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self._max_workers)

        # Chunk candidates to amortize IPC; re-score in workers
        def _score_chunk(chunk: list[tuple[str,int,int]]) -> list[Occurrence]:
            # Recompile pattern is not needed here; we’re only scoring supplied spans
            occs: list[Occurrence] = []
            for surface, s, e in chunk:
                if blacklist_context_drop(surface, text, s, e, self.cfg):
                    continue
                conf = score(surface, text, s, e, self.cfg)
                ctx = context_window(text, s, e, self.cfg.window_chars)
                occs.append(Occurrence(surface, s, e, conf, ctx))
            return occs

        futures = []
        for i in range(0, len(cands), chunk_size):
            futures.append(self._pool.submit(_score_chunk, cands[i:i+chunk_size]))

        occurrences: list[Occurrence] = []
        for f in futures:
            occurrences.extend(f.result())

        # Build first-occurrence map
        firsts: dict[str, FirstOccurrence] = {}
        for occ in occurrences:
            key = normalize_key(occ.acronym)
            if key not in firsts:
                firsts[key] = FirstOccurrence(occ.acronym, occ.start_offset, occ.end_offset, occ.confidence)
        return DetectorResult(unique_acronyms=firsts, occurrences=occurrences)

    async def detect_async(self, text: str) -> DetectorResult:
        loop = asyncio.get_running_loop()
        # Run parallel path off the event loop; won’t block FastAPI
        return await loop.run_in_executor(None, self.detect_parallel, text)



def detect_acronyms(text: str, config: DetectorConfig = DEFAULT_CONFIG) -> DetectorResult:
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

        key = normalize_key(surface)
        if key not in firsts:
            firsts[key] = FirstOccurrence(
                acronym=surface,
                start_offset=s,
                end_offset=e,
                confidence=conf,
            )

    return DetectorResult(unique_acronyms=firsts, occurrences=occurrences)
