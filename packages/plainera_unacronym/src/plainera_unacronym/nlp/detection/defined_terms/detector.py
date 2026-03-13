from __future__ import annotations

from dataclasses import replace as dc_replace

from observability.logger.decorator import logger

from plainera_unacronym.nlp.common.types import DefinedTermDetectorConfig, Span
from plainera_unacronym.nlp.detection.base import BaseDetector
from plainera_unacronym.nlp.plugins.activation import autodetect_domains

from .builders import build_defined_term_mention, build_defined_term_intro
from .compiler import compile_defined_term_patterns
from .normalise import normalize_defined_term_key
from .types import DefinedTermDetectorResult, DefinedTermMention, DefinedTermIntroduction

_QUOTE_CHARS = {'"', "“", "”"}


def _spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Return whether two half-open spans overlap.
7
    Args:
        a_start: Inclusive start offset of the first span.
        a_end: Exclusive end offset of the first span.
        b_start: Inclusive start offset of the second span.
        b_end: Exclusive end offset of the second span.

    Returns:
        True if the spans overlap by at least one character; otherwise False.
    """
    return a_start < b_end and b_start < a_end


def _overlaps_any(start: int, end: int, spans: set[Span]) -> bool:
    """Return whether a span overlaps any span in a collection.

    Args:
        start: Inclusive start offset of the candidate span.
        end: Exclusive end offset of the candidate span.
        spans: Set of existing half-open spans to compare against.

    Returns:
        True if the candidate span overlaps any span in ``spans``; otherwise
        False.
    """
    return any(_spans_overlap(start, end, s, e) for s, e in spans)


def _is_intro_term_span(start: int, end: int, spans: set[Span]) -> bool:
    return (start, end) in spans


class DefinedTermDetector(BaseDetector[DefinedTermDetectorResult]):
    """Detect defined-term introductions and occurrences in text.

    The detector identifies a bounded set of legal drafting patterns, including
    quoted term definitions, selected unquoted capitalised definitions, and later
    term occurrences. Detection is configuration-driven and may be further gated by
    automatic domain activation.

    This detector returns:
        * ``unique_terms``: canonical defined-term senses keyed by normalised term.
        * ``occurrences``: later references to previously introduced terms.
    """

    def __init__(self, config: DefinedTermDetectorConfig, max_workers=None):
        super().__init__(config=config, max_workers=max_workers)
        self._patterns = compile_defined_term_patterns()

    def _with_auto_domains(self, text: str) -> DefinedTermDetectorConfig:
        """Return config with auto-detected domains merged in.

        Args:
            text: Source text to inspect for domain activation cues.

        Returns:
            A config instance with any newly auto-detected domains merged into
            ``enabled_domains``. Returns the existing config unchanged when no new
            domains are detected.
        """
        cfg = self.cfg

        if cfg.auto_detect_domains:
            auto = autodetect_domains(text, cfg)
            if auto:
                merged = cfg.enabled_domains | auto
                if merged != cfg.enabled_domains:
                    cfg = dc_replace(cfg, enabled_domains=merged)

        legal_active = "legal" in cfg.enabled_domains

        if legal_active and not cfg.allow_unquoted_capitalised_terms:
            cfg = dc_replace(cfg, allow_unquoted_capitalised_terms=True)

        return cfg

    @staticmethod
    def _resolve_known_term_from_run(raw_term: str, known_keys: set[str]) -> tuple[str, str] | None:
        """Resolve a capitalised text run to a known defined term.

        The method first tries an exact normalised match. If that fails, it attempts
        right-trimmed suffix matching so that broader capitalised runs can resolve to a
        known defined term, for example ``"Party's Confidential Information"`` to
        ``"Confidential Information"``.

        Args:
            raw_term: Raw matched text from a capitalised occurrence pattern.
            known_keys: Set of known normalised defined-term keys introduced earlier in
                the run.

        Returns:
            A tuple of ``(resolved_term, normalised_key)`` if a known term can be
            resolved; otherwise ``None``.
        """
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
        """Extract a short definition fragment following a definition anchor.

        Extraction begins at ``anchor_end`` and stops at the earliest recognised
        boundary marker within the configured maximum character window. Leading and
        trailing spacing and lightweight punctuation noise are trimmed from the
        returned slice.

        Args:
            text: Full source text.
            anchor_end: End offset immediately after the definition anchor, such as
                ``means`` or ``shall mean``.

        Returns:
            A tuple of ``(definition, start, end)`` where ``definition`` is the trimmed
            extracted text and ``start``/``end`` are offsets into ``text`` for the
            returned slice.
        """
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
    ) -> list[DefinedTermIntroduction]:
        """Collect defined-term introductions from supported drafting patterns.

        This includes quoted introductions, selected unquoted capitalised
        introductions, and parenthetical aliases. Unquoted introductions may be gated
        by configuration and legal-domain activation.

        Args:
            text: Full source text to scan.
            cfg: Active detector configuration.
            legal_active: Whether the legal domain is currently enabled for this run.

        Returns:
            A list of ``DefinedTermIntroduction`` objects representing introduced terms in the
            order they were detected.
        """
        intros: list[DefinedTermIntroduction] = []

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
                    build_defined_term_intro(
                        term=raw_term,
                        term_start=term_start,
                        term_end=term_end,
                        provenance="defined_term_detector",
                    )
                )
        intros.sort(key=lambda intro: (intro.start_offset, intro.end_offset, intro.normalized_key))
        return intros

    def _iter_references(
        self,
        text: str,
        *,
        known_keys: set[str],
        intro_term_spans: set[Span],
        first_intro_end_by_key: dict[str, int],
        cfg: DefinedTermDetectorConfig,
        legal_active: bool,
    ) -> list[DefinedTermMention]:
        """Collect later occurrences of previously introduced defined terms.

        The detector first considers quoted occurrences and then, when allowed by
        configuration, unquoted capitalised occurrences. Unquoted matches must resolve
        back to a known term from the current run.

        Args:
            text: Full source text to scan.
            known_keys: Set of known normalised term keys introduced earlier in the
                same run.
            intro_term_spans: Spans occupied by introduction terms, used to avoid
                re-emitting introductions as occurrences.
            cfg: Active detector configuration.
            legal_active: Whether the legal domain is currently enabled for this run.

        Returns:
            A list of ``DefinedTermMention`` objects representing detected
            occurrences of known terms.
        """
        occurrences: list[DefinedTermMention] = []
        seen: set[Span] = set()

        # 1) Quoted occurrences
        for match in self._patterns.quoted_occurrence.finditer(text):
            raw_term = match.group("term")
            start_offset, end_offset = match.span("term")

            # Quoted intro terms should be suppressed, but only when the span is
            # actually the intro term span.
            if _is_intro_term_span(start_offset, end_offset, intro_term_spans):
                continue

            normalized = normalize_defined_term_key(raw_term)
            first_intro_end = first_intro_end_by_key.get(normalized)
            if first_intro_end is None or start_offset < first_intro_end:
                continue

            if normalized not in known_keys:
                continue

            span = (start_offset, end_offset)
            if span in seen:
                continue
            seen.add(span)
            occurrences.append(
                build_defined_term_mention(
                    term=raw_term,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    kind="reference",
                )
            )

        # 2) Unquoted capitalised occurrences
        if cfg.allow_unquoted_capitalised_terms and (
            (not cfg.require_legal_domain_for_unquoted) or legal_active
        ):
            for match in self._patterns.capitalised_occurrence.finditer(text):
                raw_term = match.group("term")
                start_offset, end_offset = match.span("term")

                tail = text[end_offset:].lstrip()
                if tail.startswith("("):
                    continue

                resolved = self._resolve_known_term_from_run(raw_term, known_keys)
                if not resolved:
                    continue

                resolved_term, resolved_key = resolved

                # Adjust span to the resolved suffix inside the broader match
                suffix_start = raw_term.rfind(resolved_term)
                if suffix_start == -1:
                    continue

                resolved_start = start_offset + suffix_start

                first_intro_end = first_intro_end_by_key.get(resolved_key)
                if first_intro_end is None or resolved_start < first_intro_end:
                    continue

                resolved_end = resolved_start + len(resolved_term)

                # Only now do intro suppression, against the actual resolved term span.
                if _overlaps_any(resolved_start, resolved_end, intro_term_spans):
                    continue

                span = (resolved_start, resolved_end)
                if span in seen:
                    continue
                seen.add(span)

                occurrences.append(
                    build_defined_term_mention(
                        term=resolved_term,
                        start_offset=resolved_start,
                        end_offset=resolved_end,
                        kind="reference",
                    )
                )

        return occurrences

    @logger(message="defined_term_detector.detect", db_sink="sink")
    def detect(self, text: str) -> DefinedTermDetectorResult:
        """Detect defined-term introductions and occurrences in a text run.

        The method optionally expands enabled domains through auto-detection, extracts
        term introductions, builds the unique-term index keyed by normalised term, and
        then finds later occurrences that resolve back to those introduced terms.

        Args:
            text: Full source text to analyse.

        Returns:
            A ``DefinedTermDetectorResult`` containing detected unique terms and later
            occurrences.
        """
        cfg = self._with_auto_domains(text)
        legal_active = "legal" in cfg.enabled_domains

        intros = list(self._iter_term_introductions(text, cfg, legal_active))
        first_intro_end_by_key: dict[str, int] = {}
        for intro in intros:
            prev = first_intro_end_by_key.get(intro.normalized_key)
            if prev is None or intro.end_offset < prev:
                first_intro_end_by_key[intro.normalized_key] = intro.end_offset

        unique_terms = {intro.normalized_key: intro for intro in intros}
        intro_term_spans = {(intro.start_offset, intro.end_offset) for intro in intros}

        occurrences = self._iter_references(
            text,
            known_keys=set(unique_terms.keys()),
            intro_term_spans=intro_term_spans,
            first_intro_end_by_key=first_intro_end_by_key,
            cfg=cfg,
            legal_active=legal_active,
        )

        return DefinedTermDetectorResult(
            mentions=occurrences,
            introductions=intros,
            unique_terms=unique_terms,
        )

    def detect_parallel(self, text: str, threshold: int = 1000, chunk_size: int = 256) -> DefinedTermDetectorResult:
        """Detect defined terms using the current single-pass implementation.

        This method currently delegates directly to ``detect``. The parallel entry
        point exists to preserve a stable detector interface and allow future
        structure-aware chunking if needed.

        Args:
            text: Full source text to analyse.
            threshold: Minimum text-size threshold at which a parallel strategy may be
                considered in future.
            chunk_size: Target chunk size that may be used by a future parallel
                implementation.

        Returns:
            A ``DefinedTermDetectorResult`` containing detected unique terms and later
            occurrences.
        """
        return self.detect(text)
