from __future__ import annotations

from dataclasses import replace as dc_replace

from observability.logger.decorator import logger

from plainera_unacronym.nlp.plugins.activation import autodetect_domains

from ..base import BaseDetector
from .builders import build_defined_term_occurrence, build_defined_term_sense
from .compiler import compile_defined_term_patterns
from .normalise import normalize_defined_term_key
from .types import DefinedTermDetectorResult, DefinedTermOccurrence, DefinedTermSense
from ...common.types import DefinedTermDetectorConfig, Span

_QUOTE_CHARS = {'"', "“", "”"}

def _spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def _overlaps_any(start: int, end: int, spans: set[Span]) -> bool:
    return any(_spans_overlap(start, end, s, e) for s, e in spans)


class DefinedTermDetector(BaseDetector[DefinedTermDetectorResult]):
    def __init__(self, config: DefinedTermDetectorConfig, max_workers=None):
        super().__init__(config=config, max_workers=max_workers)
        self._patterns = compile_defined_term_patterns()

    def _with_auto_domains(self, text: str) -> DefinedTermDetectorConfig:
        auto = autodetect_domains(text, self.cfg)
        if auto:
            merged = self.cfg.enabled_domains | auto
            if merged != self.cfg.enabled_domains:
                return dc_replace(self.cfg, enabled_domains=merged)
        return self.cfg

    @staticmethod
    def _resolve_known_term_from_run(raw_term: str, known_keys: set[str]) -> tuple[str, str] | None:
        parts = raw_term.split()
        if not parts:
            return None

        # Try exact first
        exact_key = normalize_defined_term_key(raw_term)
        if exact_key in known_keys:
            return raw_term, exact_key

        # Then try right-trimmed suffixes:
        # "Party's Confidential Information" -> "Confidential Information"
        for i in range(1, len(parts)):
            candidate = " ".join(parts[i:])
            key = normalize_defined_term_key(candidate)
            if key in known_keys:
                return candidate, key

        return None

    def _extract_definition_text(self, text: str, anchor_end: int) -> tuple[str, int, int]:
        raw_start = anchor_end
        raw_end = min(len(text), anchor_end + self.cfg.max_definition_chars)

        slice_text = text[raw_start:raw_end]
        stop_idx = len(slice_text)
        for marker in [".", ";", "\n"]:
            idx = slice_text.find(marker)
            if idx != -1:
                stop_idx = min(stop_idx, idx)

        raw_definition = slice_text[:stop_idx]

        trimmed_left = len(raw_definition) - len(raw_definition.lstrip(" :,-"))
        trimmed_right = len(raw_definition.rstrip(" :,-"))

        start = raw_start + trimmed_left
        end = raw_start + trimmed_right
        definition = text[start:end]

        return definition, start, end

    def _iter_term_introductions(
        self,
        text: str,
        cfg: DefinedTermDetectorConfig,
        legal_active: bool,
    ) -> list[DefinedTermSense]:
        intros: list[DefinedTermSense] = []

        for pat_name in (
                "quoted_means",
                "quoted_shall_mean",
                "bare_means",
                "bare_shall_mean",
                "parenthetical_alias",
        ):
            pat = getattr(self._patterns, pat_name)
            for match in pat.finditer(text):
                group_name = "term_q" if match.groupdict().get("term_q") else "term_b"
                raw_term = match.group(group_name)
                if not raw_term:
                    continue

                term_start, term_end = match.span(group_name)
                if raw_term[:1] in {'"', "“", "”"} and raw_term[-1:] in {'"', "“", "”"}:
                    term_start += 1
                    term_end -= 1

                is_quoted = raw_term.startswith('"') and raw_term.endswith('"')
                if not is_quoted:
                    if cfg.require_legal_domain_for_unquoted and not legal_active:
                        continue
                    if not cfg.allow_unquoted_capitalised_terms:
                        continue

                intros.append(
                    build_defined_term_sense(
                        term=raw_term,
                        term_start=term_start,
                        term_end=term_end,
                        provenance="defined_term_detector",
                    )
                )

        return intros

    def _iter_occurrences(
        self,
        text: str,
        *,
        known_keys: set[str],
        intro_term_spans: set[Span],
        cfg: DefinedTermDetectorConfig,
        legal_active: bool,
    ) -> list[DefinedTermOccurrence]:
        occurrences: list[DefinedTermOccurrence] = []

        # 1) Quoted occurrences
        for match in self._patterns.quoted_occurrence.finditer(text):
            raw_term = match.group("term")
            start_offset, end_offset = match.span("term")
            if _overlaps_any(start_offset, end_offset, intro_term_spans):
                continue

            normalized = normalize_defined_term_key(raw_term)
            if normalized not in known_keys:
                continue
            occurrences.append(
                build_defined_term_occurrence(
                    term=raw_term,
                    start_offset=start_offset,
                    end_offset=end_offset,
                )
            )

        # 2) Unquoted capitalised occurrences
        if cfg.allow_unquoted_capitalised_terms and (
            (not cfg.require_legal_domain_for_unquoted) or legal_active
        ):
            for match in self._patterns.capitalised_occurrence.finditer(text):
                raw_term = match.group("term")
                start_offset, end_offset = match.span("term")

                if _overlaps_any(start_offset, end_offset, intro_term_spans):
                    continue

                tail = text[end_offset:].lstrip()
                if tail.startswith("("):
                    continue

                resolved = self._resolve_known_term_from_run(raw_term, known_keys)
                if not resolved:
                    continue

                resolved_term, _ = resolved

                # Adjust span to the resolved suffix inside the broader match
                suffix_start = raw_term.rfind(resolved_term)
                if suffix_start == -1:
                    continue

                resolved_start = start_offset + suffix_start
                resolved_end = resolved_start + len(resolved_term)

                occurrences.append(
                    build_defined_term_occurrence(
                        term=resolved_term,
                        start_offset=resolved_start,
                        end_offset=resolved_end,
                    )
                )

        return occurrences

    @logger(message="defined_term_detector.detect", db_sink="sink")
    def detect(self, text: str) -> DefinedTermDetectorResult:
        cfg = self._with_auto_domains(text)
        legal_active = "legal" in cfg.enabled_domains

        intros = self._iter_term_introductions(text, cfg, legal_active)
        unique_terms = {intro.normalized_key: intro for intro in intros}
        intro_term_spans = {(intro.start_offset, intro.end_offset) for intro in intros}

        occurrences = self._iter_occurrences(
            text,
            known_keys=set(unique_terms.keys()),
            intro_term_spans=intro_term_spans,
            cfg=cfg,
            legal_active=legal_active,
        )

        return DefinedTermDetectorResult(
            occurrences=occurrences,
            unique_terms=unique_terms,
        )

    def detect_parallel(self, text: str, threshold: int = 1000, chunk_size: int = 256) -> DefinedTermDetectorResult:
        # Fine to keep this simple initially.
        # Defined-term detection is more structure-sensitive than acronym scanning.
        return self.detect(text)
