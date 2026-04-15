import re
from collections.abc import Iterator

from document_resolution.nlp.common.types import Span, TextSpanTuple

from .config import _STATS_CI_RE, _STATS_OR_HR_RR_RE, BioConfig
from .patterns import bio_pattern


def extra_candidates(text: str, cfg: BioConfig) -> Iterator[TextSpanTuple]:
    """Yield bio-domain candidate spans found by bio-specific patterns.

    Runs the domain regex to capture biomedical tokens that the generic detector
    may miss (e.g. cytokines/viruses/UTRs), then optionally adds explicit RNA-like
    terms from `cfg.rna_like`. This yields raw spans only; downstream gates decide
    acceptance.

    Args:
        text (str): Source text to scan.
        cfg (BioConfig): Bio configuration (uses `rna_like` to add explicit tokens).

    Yields:
        TextSpanTuple: (surface, start, end) matches with end-exclusive offsets.
    """
    pat = bio_pattern()
    for m in pat.finditer(text):
        s, e = m.span("bio")
        yield text[s:e], s, e

    # 2) Explicit RNA-like tokens (captures mRNA/miRNA/sgRNA that start lowercase)
    if cfg.rna_like:
        rna_re = re.compile(r"\b(?:" + "|".join(map(re.escape, cfg.rna_like)) + r")\b")
        for m in rna_re.finditer(text):
            yield m.group(0), m.start(), m.end()


def _sentence_slice(text: str, s: int, e: int, max_chars: int) -> Span:
    """Return a bounded sentence-like slice around a target span.

    Expands to nearest sentence terminators around (s, e) and then clamps the
    slice to `max_chars` around the midpoint to avoid pathological long sentences.

    Args:
        text (str): Source text.
        s (int): Start offset (inclusive) of the target span.
        e (int): End offset (exclusive) of the target span.
        max_chars (int): Maximum slice width after clamping.

    Returns:
        Span: (start, end) offsets delimiting the slice (end-exclusive).
    """
    left = max(text.rfind(".", 0, s), text.rfind("?", 0, s), text.rfind("!", 0, s))
    right_candidates = [text.find(".", e), text.find("?", e), text.find("!", e)]
    right = min([p for p in right_candidates if p != -1] or [len(text)])
    a, b = left + 1 if left != -1 else 0, right
    # soft clamp for pathological sentences
    if b - a > max_chars:
        mid = (s + e) // 2
        half = max_chars // 2
        a = max(a, mid - half)
        b = min(b, mid + half)
    return a, b


def bio_keep_guard(surface: str, text: str, s: int, e: int, cfg: BioConfig) -> bool:
    """Domain-specific rescue/keep rule for borderline biomedical candidates.

    Keeps known RNA-like tokens unconditionally, and conditionally keeps certain
    ambiguous two-letter tokens (e.g. OR/HR/RR) only when local sentence context
    contains statistical markers such as CI/OR/HR/RR patterns.

    Args:
        surface (str): Candidate surface text (`text[s:e]`).
        text (str): Full source text.
        s (int): Start offset (inclusive).
        e (int): End offset (exclusive).
        cfg (BioConfig): Bio configuration (uses `rna_like`, `two_letter_keep`, and `stats_window_chars`).

    Returns:
        bool: True if the token should be kept by the bio domain; otherwise False.
    """
    if surface in cfg.rna_like:
        return True

    if len(surface) == 2 and surface in cfg.two_letter_keep and surface.isupper():
        a, b = _sentence_slice(text, s, e, cfg.stats_window_chars or 60)
        r = text[a:b]
        return bool(_STATS_CI_RE.search(r) or _STATS_OR_HR_RR_RE.search(r))

    return False
