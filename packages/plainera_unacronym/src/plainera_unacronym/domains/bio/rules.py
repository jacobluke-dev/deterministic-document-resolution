import re
from typing import Iterator, Tuple

from .config import _STATS_CI_RE, _STATS_OR_HR_RR_RE, BioConfig
from .patterns import bio_pattern

Span = Tuple[str, int, int]


def extra_candidates(text: str, cfg: BioConfig) -> Iterator[Span]:
    """Yield biomedical candidate spans found by the bio regex.

    Scans ``text`` with the precompiled bio pattern (see ``bio_pattern()``) to
    identify domain-specific tokens such as cytokines (e.g., ``IL-6``,
    ``TNF-α``, ``IFN-γ``, ``TGF-β1``), viral names (e.g., ``SARS-CoV-2``,
    ``MERS-CoV``, ``H1N1``), UTR markers (``5′-UTR`` / ``3′-UTR``), and
    gene/protein-like camel-case forms (e.g., ``BRCA1``). This function only
    proposes raw spans; downstream guards should decide whether to keep them.

    Args:
      text: Source text to scan.
      cfg: Bio domain configuration. Accepted for interface symmetry and
        potential future tuning; not currently read by this function.

    Yields:
      Span: Tuples of ``(surface: str, start: int, end: int)`` for each match,
      where ``start`` (inclusive) and ``end`` (exclusive) are character offsets
      into ``text``.

    Example:
      >>> list(extra_candidates("Measured IL-6 and IFN-γ in SARS-CoV-2 samples.", BioConfig()))
      [('IL-6', 9, 13), ('IFN-γ', 18, 22), ('SARS-CoV-2', 26, 36)]
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


def _sentence_slice(text: str, s: int, e: int, max_chars: int) -> tuple[int, int]:
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


def keep_guard(surface, text, s, e, cfg: BioConfig) -> bool:
    if surface in cfg.rna_like:
        return True
    if len(surface) == 2 and surface in cfg.two_letter_keep and surface.isupper():
        a, b = _sentence_slice(text, s, e, cfg.stats_window_chars or 60)
        r = text[a:b]
        return bool(_STATS_CI_RE.search(r) or _STATS_OR_HR_RR_RE.search(r))
    return False
