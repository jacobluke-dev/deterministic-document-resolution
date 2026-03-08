from dataclasses import replace as dc_replace

from observability.logger.decorator import logger

from plainera_unacronym.nlp.plugins.activation import autodetect_domains

from ..base import BaseDetector
from .builders import build_defined_term_occurrence, build_defined_term_sense
from .compiler import compile_defined_term_patterns
from .normalise import normalize_defined_term_key
from .types import DefinedTermDetectorResult, DefinedTermOccurrence, DefinedTermSense
from ...common.types import DefinedTermDetectorConfig


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

    def _iter_senses(self, text: str, cfg: DefinedTermDetectorConfig, legal_active: bool) -> list[DefinedTermSense]:
        senses: list[DefinedTermSense] = []

        for pat_name in ("quoted_means", "quoted_shall_mean", "bare_means", "bare_shall_mean"):
            pat = getattr(self._patterns, pat_name)
            for match in pat.finditer(text):
                group_name = "term_q" if match.groupdict().get("term_q") else "term_b"
                raw_term = match.group(group_name)
                if not raw_term:
                    continue

                term_start, term_end = match.span(group_name)

                is_quoted = raw_term.startswith('"') and raw_term.endswith('"')
                if not is_quoted:
                    if cfg.require_legal_domain_for_unquoted and not legal_active:
                        continue
                    if not cfg.allow_unquoted_capitalised_terms:
                        continue

                definition_text, def_start, def_end = self._extract_definition_text(text, match.end())
                if not definition_text:
                    continue

                senses.append(
                    build_defined_term_sense(
                        term=raw_term,
                        term_start=term_start,
                        term_end=term_end,
                        definition_text=definition_text,
                        definition_start=def_start,
                        definition_end=def_end,
                        provenance="defined_term_detector",
                    )
                )

        return senses

    def _iter_occurrences(
        self,
        text: str,
        *,
        known_keys: set[str],
        cfg: DefinedTermDetectorConfig,
        legal_active: bool,
    ) -> list[DefinedTermOccurrence]:
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

        if (not cfg.require_legal_domain_for_unquoted) or legal_active:
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
        cfg = self._with_auto_domains(text)
        legal_active = "legal" in cfg.enabled_domains

        senses = self._iter_senses(text, cfg, legal_active)
        unique_terms = {sense.normalized_key: sense for sense in senses}
        occurrences = self._iter_occurrences(text,
                                             known_keys=set(unique_terms.keys()),
                                             cfg=cfg,
                                             legal_active=legal_active)

        return DefinedTermDetectorResult(
            senses=senses,
            occurrences=occurrences,
            unique_terms=unique_terms,
        )

    def detect_parallel(self, text: str, threshold: int = 1000, chunk_size: int = 256) -> DefinedTermDetectorResult:
        # Fine to keep this simple initially.
        # Defined-term detection is more structure-sensitive than acronym scanning.
        return self.detect(text)
