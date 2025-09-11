from concurrent.futures import ProcessPoolExecutor
import asyncio
from typing import Optional

from plainera_unacronym.nlp.config import ALLOW_CHARS, DOT_MODE
from plainera_unacronym.nlp.heuristics.core import score, threshold_len, normalize_key, context_window, compile_pattern, \
    iter_candidates_with, iter_candidates, reason_tags
from plainera_unacronym.nlp.heuristics.general import blacklist_context_drop, strip_terminal_plural
from plainera_unacronym.nlp.types import DetectorConfig, DetectorResult, Occurrence, FirstOccurrence

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


def _score_chunk_worker(cfg: DetectorConfig, text: str, window_chars: int, cands: list[tuple[str,int,int]]):
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

            occ, display_key = _build_occurrence_from_match(self.cfg, text, surface, s, e, conf)
            occurrences.append(occ)

            if display_key not in firsts:
                firsts[display_key] = FirstOccurrence(
                    acronym=occ.acronym,
                    start_offset=occ.start_offset,
                    end_offset=occ.end_offset,
                    confidence=conf,
                    normalized_key=display_key,  # if present in the dataclass
                )

        return DetectorResult(unique_acronyms=firsts, occurrences=occurrences)

    def detect_parallel(self, text: str, threshold: int = 1000, chunk_size: int = 256) -> DetectorResult:
        cands = list(iter_candidates_with(text, self.cfg, self._pat))
        if len(cands) < threshold:
            return self.detect(text)

        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self._max_workers)

        futures = []
        for i in range(0, len(cands), chunk_size):
            futures.append(
                self._pool.submit(_score_chunk_worker, self.cfg, text, self.cfg.window_chars, cands[i:i + chunk_size]))

        occurrences: list[Occurrence] = []
        for f in futures:
            occurrences.extend(f.result())

        firsts: dict[str, FirstOccurrence] = {}
        # prefer the normalized_key computed in the helper if available
        for occ in occurrences:
            if not isinstance(occ.acronym, str):
                raise TypeError(f"Occurrence.acronym is not str: {type(occ.acronym)} -> {occ}")

            display_key = getattr(occ, "normalized_key", normalize_key(
                occ.acronym,
                self.cfg.allow_chars,
                dotted_mode=getattr(self.cfg, "dotted_display", "strip"),
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
        loop = asyncio.get_running_loop()
        # Run parallel path off the event loop; won’t block FastAPI
        return await loop.run_in_executor(None, self.detect_parallel, text)


def detect_acronyms(text: str,
                    config: DetectorConfig = DEFAULT_CONFIG,
                    allowed_chars=ALLOW_CHARS_DEFAULTS,
                    dot_mode=DEFAULT_DOT_MODE) -> DetectorResult:
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

        key = normalize_key(surface, allowed_chars, dot_mode)
        if key not in firsts:
            firsts[key] = FirstOccurrence(
                acronym=surface,
                start_offset=s,
                end_offset=e,
                confidence=conf,
            )

    return DetectorResult(unique_acronyms=firsts, occurrences=occurrences)
