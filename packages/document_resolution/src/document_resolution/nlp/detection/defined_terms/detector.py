from __future__ import annotations

import re
from dataclasses import replace as dc_replace

from observability.logger.decorator import logger

from document_resolution.nlp.common.types import DefinedTermDetectorConfig, Span
from document_resolution.nlp.detection.base import BaseDetector
from document_resolution.nlp.plugins.activation import autodetect_domains

from .builders import build_defined_term_intro, build_defined_term_mention
from .compiler import compile_defined_term_patterns
from .normalise import normalize_defined_term_key
from .types import DefinedTermDetectorResult, DefinedTermIntroduction, DefinedTermMention, IntroKind

_QUOTE_CHARS = {'"', "“", "”"}


def _is_word_char(ch: str) -> bool:
    """Return whether a character should count as part of a word token."""
    return ch.isalnum() or ch == "_"


def _has_token_boundaries(text: str, start: int, end: int) -> bool:
    """Return whether ``text[start:end]`` is bounded by non-word characters."""
    if start > 0 and _is_word_char(text[start - 1]):
        return False

    has_trailing_word_char = end < len(text) and _is_word_char(text[end])
    return not has_trailing_word_char


def _term_surface_variants(term: str) -> list[str]:
    variants = [term]

    if term.endswith("y") and len(term) > 1 and term[-2].lower() not in "aeiou":
        variants.append(f"{term[:-1]}ies")
    elif term.endswith(("s", "x", "z", "ch", "sh")):
        variants.append(f"{term}es")
    else:
        variants.append(f"{term}s")

    seen: set[str] = set()
    ordered: list[str] = []
    for variant in variants:
        if variant not in seen:
            seen.add(variant)
            ordered.append(variant)
    return ordered


def _spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Return whether two half-open spans overlap.
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

    This detector returns:
        * ``unique_terms``: canonical defined-term meanings keyed by normalised term.
        * ``occurrences``: later references to previously introduced terms.
    """

    def __init__(
        self,
        config: DefinedTermDetectorConfig,
        max_workers: int | None = None,
        sink=None,
    ):
        super().__init__(config=config, max_workers=max_workers, sink=sink)
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

        return cfg

    @staticmethod
    def _resolve_known_term_from_run(raw_term: str, known_keys: set[str]) -> tuple[str, str] | None:
        """Resolve a capitalised text run to a known defined term.

        The method first tries an exact normalised match. If that fails, it attempts to
        recover a known term from either edge of a broader capitalised run:

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

        # Try left-trimmed prefixes so a broader capitalised run can resolve to a known
        # term at the start, e.g. "Services to the Customer" -> "Services".
        for i in range(len(parts) - 1, 0, -1):
            candidate = " ".join(parts[:i])
            key = normalize_defined_term_key(candidate)
            if key in known_keys:
                return candidate, key

        return None

    @staticmethod
    def _iter_exact_known_term_references(
        text: str,
        *,
        introductions: list[DefinedTermIntroduction],
        intro_term_spans: set[Span],
        first_intro_end_by_key: dict[str, int],
        seen: set[Span],
    ) -> list[DefinedTermMention]:
        """Collect exact later references to introduced defined terms.

        This pass performs a literal scan for each introduced term surface after
        its first introduction. It is intended to catch straightforward exact
        later mentions such as ``Effective Date`` that may be missed by the more
        generic capitalised-run patterns.

        Args:
            text: Full source text to scan.
            introductions: Introduced defined terms from the current run.
            intro_term_spans: Exact introduction-term spans used to suppress
                re-emitting introduction text as later references.
            first_intro_end_by_key: Mapping from normalised key to the end offset
                of the earliest introduction term span.
            seen: Set of already-emitted spans used for deduplication across all
                reference passes.

        Returns:
            A list of exact-match ``DefinedTermMention`` objects detected in
            document order.
        """
        occurrences: list[DefinedTermMention] = []
        seen_keys: set[str] = set()

        for intro in introductions:
            if intro.normalized_key in seen_keys:
                continue
            seen_keys.add(intro.normalized_key)

            first_intro_end = first_intro_end_by_key.get(intro.normalized_key)
            if first_intro_end is None:
                continue

            for variant in _term_surface_variants(intro.term):
                pattern = re.compile(re.escape(variant))

                for match in pattern.finditer(text):
                    start_offset, end_offset = match.span()
                    matched_text = match.group(0)

                    if start_offset < first_intro_end:
                        continue
                    if not _has_token_boundaries(text, start_offset, end_offset):
                        continue
                    if _overlaps_any(start_offset, end_offset, intro_term_spans):
                        continue

                    span = (start_offset, end_offset)
                    if span in seen:
                        continue
                    seen.add(span)

                    occurrences.append(
                        DefinedTermMention(
                            term=matched_text,
                            start_offset=start_offset,
                            end_offset=end_offset,
                            normalized_key=intro.normalized_key,
                            confidence=1.0,
                            segment_window=None,
                        )
                    )

        return occurrences

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
            A list of ``DefinedTermIntroduction`` objects representing introduced terms
            in the order they were detected.
        """
        intros: list[DefinedTermIntroduction] = []
        intro_pattern_names: tuple[IntroKind, ...] = (
            "quoted_means",
            "quoted_shall_mean",
            "bare_means",
            "bare_shall_mean",
            "parenthetical_alias",
        )

        policy = cfg.unquoted_capitalised_terms_policy
        allow_unquoted = policy == "always" or (policy == "legal_only" and legal_active)

        for pat_name in intro_pattern_names:
            pat = getattr(self._patterns, pat_name)
            for match in pat.finditer(text):
                group_name = "term_q" if match.groupdict().get("term_q") else "term_b"
                raw_term = match.group(group_name)
                if not raw_term:
                    continue

                term_start, term_end = match.span(group_name)
                if raw_term[:1] in _QUOTE_CHARS and raw_term[-1:] in _QUOTE_CHARS:
                    term_start += 1
                    term_end -= 1

                is_quoted = raw_term[:1] in _QUOTE_CHARS and raw_term[-1:] in _QUOTE_CHARS
                if not is_quoted and not allow_unquoted:
                    continue

                intros.append(
                    build_defined_term_intro(
                        term=raw_term,
                        term_start=term_start,
                        term_end=term_end,
                        provenance="defined_term_detector",
                        intro_kind=pat_name,
                    )
                )
        intros.sort(key=lambda intro: (intro.start_offset, intro.end_offset, intro.normalized_key))
        return intros

    @staticmethod
    def _append_reference_if_new(
        occurrences: list[DefinedTermMention],
        seen: set[Span],
        *,
        term: str,
        start_offset: int,
        end_offset: int,
    ) -> None:
        """Append a detected reference span if it has not already been emitted.

        Uses the exact ``(start_offset, end_offset)`` span as the deduplication key.
        When the span is new, a ``DefinedTermMention`` is built and appended to the
        output list and the span is recorded in ``seen``.

        Args:
            occurrences: Accumulated output list of detected later references.
            seen: Set of already-emitted spans used for deduplication.
            term: Cleaned or resolved defined-term surface text to emit.
            start_offset: Inclusive start offset of the reference span.
            end_offset: Exclusive end offset of the reference span.
        """
        span = (start_offset, end_offset)
        if span in seen:
            return
        seen.add(span)
        occurrences.append(
            build_defined_term_mention(
                term=term,
                start_offset=start_offset,
                end_offset=end_offset,
            )
        )

    def _iter_quoted_references(
        self,
        text: str,
        *,
        known_keys: set[str],
        intro_term_spans: set[Span],
        first_intro_end_by_key: dict[str, int],
        seen: set[Span],
    ) -> list[DefinedTermMention]:
        """Collect quoted later references to previously introduced defined terms.

        This pass scans quoted occurrence patterns, suppresses exact introduction-term
        spans, ignores references that occur before the first introduction for a key,
        and only emits terms already known from the current run.

        Args:
            text: Full source text to scan.
            known_keys: Set of known normalised defined-term keys introduced earlier in
                the same run.
            intro_term_spans: Exact spans occupied by introduction terms, used to
                suppress re-emitting introductions as later references.
            first_intro_end_by_key: Mapping from normalised key to the end offset of
                its earliest introduction term span.
            seen: Set of already-emitted spans used for deduplication across quoted
                and unquoted reference passes.

        Returns:
            A list of quoted ``DefinedTermMention`` objects detected in document order.
        """
        occurrences: list[DefinedTermMention] = []

        for match in self._patterns.quoted_occurrence.finditer(text):
            raw_term = match.group("term")
            start_offset, end_offset = match.span("term")

            if _is_intro_term_span(start_offset, end_offset, intro_term_spans):
                continue

            normalized = normalize_defined_term_key(raw_term)
            first_intro_end = first_intro_end_by_key.get(normalized)
            if first_intro_end is None or start_offset < first_intro_end:
                continue

            if normalized not in known_keys:
                continue

            self._append_reference_if_new(
                occurrences,
                seen,
                term=raw_term,
                start_offset=start_offset,
                end_offset=end_offset,
            )

        return occurrences

    def _iter_unquoted_references(
        self,
        text: str,
        *,
        known_keys: set[str],
        intro_term_spans: set[Span],
        first_intro_end_by_key: dict[str, int],
        cfg: DefinedTermDetectorConfig,
        legal_active: bool,
        seen: set[Span],
    ) -> list[DefinedTermMention]:
        """Collect unquoted capitalised later references to introduced defined terms.

        This pass is only active when unquoted-capitalised reference handling is
        enabled by configuration and, when required, the legal domain is active.
        Broader capitalised runs are resolved back to known term keys using suffix
        or prefix matching, for example resolving
        ``"Party's Confidential Information"`` to ``"Confidential Information"``
        or ``"Services to the Customer"`` to ``"Services"``.
        Args:
            text: Full source text to scan.
            known_keys: Set of known normalised defined-term keys introduced earlier in
                the same run.
            intro_term_spans: Exact introduction-term spans used to suppress
                re-emitting introduction text as later references.
            first_intro_end_by_key: Mapping from normalised key to the end offset of
                its earliest introduction term span.
            cfg: Active detector configuration controlling unquoted reference behaviour.
            legal_active: Whether the legal domain is currently enabled for this run.
            seen: Set of already-emitted spans used for deduplication across quoted
                and unquoted reference passes.

        Returns:
            A list of unquoted ``DefinedTermMention`` objects detected in document
            order.
        """
        occurrences: list[DefinedTermMention] = []

        policy = cfg.unquoted_capitalised_terms_policy
        effective_allow_unquoted = policy == "always" or (policy == "legal_only" and legal_active)

        if not effective_allow_unquoted:
            return occurrences

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

            resolved_rel_start = raw_term.find(resolved_term)
            if resolved_rel_start == -1:
                continue

            resolved_start = start_offset + resolved_rel_start
            resolved_end = resolved_start + len(resolved_term)

            first_intro_end = first_intro_end_by_key.get(resolved_key)
            if first_intro_end is None or resolved_start < first_intro_end:
                continue

            if _overlaps_any(resolved_start, resolved_end, intro_term_spans):
                continue

            self._append_reference_if_new(
                occurrences,
                seen,
                term=resolved_term,
                start_offset=resolved_start,
                end_offset=resolved_end,
            )
        return occurrences

    def _iter_references(
        self,
        text: str,
        *,
        introductions: list[DefinedTermIntroduction],
        known_keys: set[str],
        intro_term_spans: set[Span],
        first_intro_end_by_key: dict[str, int],
        cfg: DefinedTermDetectorConfig,
        legal_active: bool,
    ) -> list[DefinedTermMention]:
        """Collect later references to previously introduced defined terms.

        Runs an exact known-term pass first, followed by the quoted-reference
        pass and then the unquoted capitalised reference pass. All passes share a
        single deduplication set so the same span is not emitted twice.

        Args:
            text: Full source text to scan.
            introductions: Introduced defined terms from the current run.
            known_keys: Set of known normalised defined-term keys introduced earlier in
                the same run.
            intro_term_spans: Exact spans occupied by introduction terms, used to
                suppress re-emitting introductions as later references.
            first_intro_end_by_key: Mapping from normalised key to the end offset of
                its earliest introduction term span.
            cfg: Active detector configuration.
            legal_active: Whether the legal domain is currently enabled for this run.

        Returns:
            A list of ``DefinedTermMention`` objects representing later references to
            known terms.
        """
        occurrences: list[DefinedTermMention] = []
        seen: set[Span] = set()

        occurrences.extend(
            self._iter_exact_known_term_references(
                text,
                introductions=introductions,
                intro_term_spans=intro_term_spans,
                first_intro_end_by_key=first_intro_end_by_key,
                seen=seen,
            )
        )

        occurrences.extend(
            self._iter_quoted_references(
                text,
                known_keys=known_keys,
                intro_term_spans=intro_term_spans,
                first_intro_end_by_key=first_intro_end_by_key,
                seen=seen,
            )
        )

        occurrences.extend(
            self._iter_unquoted_references(
                text,
                known_keys=known_keys,
                intro_term_spans=intro_term_spans,
                first_intro_end_by_key=first_intro_end_by_key,
                cfg=cfg,
                legal_active=legal_active,
                seen=seen,
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

        unique_terms: dict[str, DefinedTermIntroduction] = {}
        for intro in intros:
            unique_terms.setdefault(intro.normalized_key, intro)
        intro_term_spans = {(intro.start_offset, intro.end_offset) for intro in intros}

        intro_mentions = [
            build_defined_term_mention(
                term=intro.term,
                start_offset=intro.start_offset,
                end_offset=intro.end_offset,
            )
            for intro in intros
        ]

        later_mentions = self._iter_references(
            text,
            introductions=intros,
            known_keys=set(unique_terms.keys()),
            intro_term_spans=intro_term_spans,
            first_intro_end_by_key=first_intro_end_by_key,
            cfg=cfg,
            legal_active=legal_active,
        )

        occurrences = sorted(
            [*intro_mentions, *later_mentions],
            key=lambda occ: (occ.start_offset, occ.end_offset, occ.normalized_key),
        )

        return DefinedTermDetectorResult(
            mentions=occurrences,
            introductions=intros,
            unique_terms=unique_terms,
        )

    def detect_parallel(self, text: str, threshold: int = 1000, chunk_size: int = 256) -> DefinedTermDetectorResult:
        """This method currently calls ``detect``. The parallel entry
        point exists to preserve a stable detector interface and allow future
        structure-aware chunking if needed.

        Args:
            text: Full source text to analyse
            threshold: Minimum text-size threshold
            chunk_size: Target chunk size

        Returns:
            A ``DefinedTermDetectorResult`` containing detected unique terms and later
            occurrences.
        """
        return self.detect(text)
