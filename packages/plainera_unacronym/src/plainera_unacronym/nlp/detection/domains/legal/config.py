import re
from dataclasses import dataclass
from typing import Optional

from plainera_unacronym.nlp.common.types import DetectorConfig

# Strong legal cues (keep conservative to avoid false positives)
_LEGAL_MEANS_RE = re.compile(r'\b"[^"]{1,80}"\s+(?:shall\s+)?mean(?:s)?\b', re.IGNORECASE)
_AGREEMENT_CUE_RE = re.compile(r"\bthis\s+Agreement\b", re.IGNORECASE)
_SCHEDULE_CUE_RE = re.compile(r"\b(?:Schedule|Appendix|Exhibit)\s+[A-Z0-9]+\b", re.IGNORECASE)
_CLAUSE_SECTION_RE = re.compile(r"\b(?:clause|section)\s+\d+(?:\.\d+)*\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class LegalConfig(DetectorConfig):
    enable_legal: bool = False

    # Sniff tuning: require N hits within cap window
    sniff_cap: int = 80_000
    sniff_min_hits: int = 1  # start at 1; raise to 2 if you see false positives

    # Future: defined-term related (safe to add now, unused in UN-86)
    quoted_terms_only: bool = True
    term_context_window_chars: Optional[int] = 200
