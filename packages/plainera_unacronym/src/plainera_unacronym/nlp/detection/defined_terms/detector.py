from dataclasses import dataclass

from observability.logger.decorator import logger

from plainera_unacronym.nlp.plugins.activation import autodetect_domains

from ..base import BaseDetector
from .builders import build_defined_term_occurrence, build_defined_term_sense
from .compiler import compile_defined_term_patterns
from .normalise import normalize_defined_term_key
from .types import DefinedTermDetectorResult, DefinedTermOccurrence, DefinedTermSense


@dataclass(frozen=True)
class DefinedTermDetectorConfig:
    window_chars: int = 80
    allow_unquoted_capitalised_terms: bool = False
    require_legal_domain_for_unquoted: bool = True
    max_definition_chars: int = 500


class DefinedTermDetector(BaseDetector[DefinedTermDetectorResult]):
    def __init__(self, config: DefinedTermDetectorConfig, max_workers=None):
        super().__init__(config=config, max_workers=max_workers)
        self._patterns = compile_defined_term_patterns()

    def _legal_domain_active(self, text: str) -> bool:
        auto = autodetect_domains(text, self.cfg)
        return "legal" in auto

    def _extract_definition_text(self, text: str, anchor_end: int) -> tuple[str, int, int]:
        start = anchor_end
        end = min(len(text), anchor_end + self.cfg.max_definition_chars)

        slice_text = text[start:end]
        stop_idx = len(slice_text)
        for marker in [".", ";", "\n"]:
            idx = slice_text.find(marker)
            if idx != -1:
                stop_idx = min(stop_idx, idx)

        definition = slice_text[:stop_idx].strip(" :,-")
        return definition, start, start + len(definition)

    def _iter_senses(self, text: str) -> list[DefinedTermSense]:
        senses: list[DefinedTermSense] = []

        for pat_name in ("quoted_means", "quoted_shall_mean", "bare_means", "bare_shall_mean"):
            pat = getattr(self._patterns, pat_name)
            for match in pat.finditer(text):
                raw_term = match.groupdict().get("term_q") or match.groupdict().get("term_b")
                if not raw_term:
                    continue

                is_quoted = raw_term.startswith('"') and raw_term.endswith('"')
                if not is_quoted:
                    if self.cfg.require_legal_domain_for_unquoted and not self._legal_domain_active(text):
                        continue
                    if not self.cfg.allow_unquoted_capitalised_terms:
                        continue

                definition_text, def_start, def_end = self._extract_definition_text(text, match.end())
                if not definition_text:
                    continue

                senses.append(
                    build_defined_term_sense(
                        term=raw_term,
                        term_start=match.start(),
                        term_end=match.end(raw_term.strip('"')) if False else match.start() + len(raw_term),
                        definition_text=definition_text,
                        definition_start=def_start,
                        definition_end=def_end,
                        provenance="defined_term_detector",
                    )
                )

        return senses

    def _iter_occurrences(self, text: str, known_keys: set[str]) -> list[DefinedTermOccurrence]:
        occurrences: list[DefinedTermOccurrence] = []

        for match in self._patterns.quoted_occurrence.finditer(text):
            raw_term = match.group("term")
            normalized = normalize_defined_term_key(raw_term)
            if normalized not in known_keys:
                continue

            occurrences.append(
                build_defined_term_occurrence(
                    term=raw_term,
                    start_offset=match.start(),
                    end_offset=match.end(),
                )
            )

        if self.cfg.allow_unquoted_capitalised_terms and self._legal_domain_active(text):
            for match in self._patterns.capitalised_occurrence.finditer(text):
                raw_term = match.group("term")
                normalized = normalize_defined_term_key(raw_term)
                if normalized not in known_keys:
                    continue

                occurrences.append(
                    build_defined_term_occurrence(
                        term=raw_term,
                        start_offset=match.start(),
                        end_offset=match.end(),
                    )
                )

        return occurrences

    @logger(message="defined_term_detector.detect", db_sink="sink")
    def detect(self, text: str) -> DefinedTermDetectorResult:
        senses = self._iter_senses(text)
        unique_terms = {sense.normalized_key: sense for sense in senses}
        occurrences = self._iter_occurrences(text, known_keys=set(unique_terms.keys()))

        return DefinedTermDetectorResult(
            senses=senses,
            occurrences=occurrences,
            unique_terms=unique_terms,
        )

    def detect_parallel(self, text: str, threshold: int = 1000, chunk_size: int = 256) -> DefinedTermDetectorResult:
        # Fine to keep this simple initially.
        # Defined-term detection is more structure-sensitive than acronym scanning.
        return self.detect(text)
