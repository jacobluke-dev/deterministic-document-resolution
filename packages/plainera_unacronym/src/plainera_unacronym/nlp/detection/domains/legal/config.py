import re
from dataclasses import dataclass
from typing import Optional

from plainera_unacronym.nlp.common.types import DetectorConfig

# STRONG (high precision)
LEGAL_QUOTED_MEANS_RE = re.compile(r"\"[A-Z][^\"]{1,80}\"\s+(?:shall\s+)?mean(?:s)?\b", re.IGNORECASE)
LEGAL_HEREINAFTER_RE = re.compile(r"\bhereinafter\b", re.IGNORECASE)
LEGAL_THIS_AGREEMENT_RE = re.compile(r"\bthis\s+Agreement\b", re.IGNORECASE)

# SUPPORT (contextual)
LEGAL_PURSUANT_RE = re.compile(r"\bpursuant\s+to\b", re.IGNORECASE)
LEGAL_SCHEDULE_RE = re.compile(r"\b(?:Schedule|Appendix|Exhibit)\s+[A-Z0-9]+\b", re.IGNORECASE)
LEGAL_CLAUSE_SECTION_RE = re.compile(r"\b(?:clause|section)\s+\d+(?:\.\d+)*\b", re.IGNORECASE)
LEGAL_GOV_LAW_RE = re.compile(r"\bgoverning\s+law\b|\bjurisdiction\b", re.IGNORECASE)
LEGAL_INDEMNITY_RE = re.compile(r"\bindemnif(?:y|ication)\b|\bliabilit(?:y|ies)\b", re.IGNORECASE)
LEGAL_FORCE_MAJEURE_RE = re.compile(r"\bforce\s+majeure\b", re.IGNORECASE)
LEGAL_EU_CITE_RE = re.compile(r"\bRegulation\s*\(\s*EU\s*\)\s*\d{3,4}/\d{2,4}\b", re.IGNORECASE)
LEGAL_ACT_YEAR_RE = re.compile(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4}\s+Act\s+\d{4}\b")

LEGAL_STRONG_CUES = (
    ("quoted_means", LEGAL_QUOTED_MEANS_RE, 6),
    ("hereinafter", LEGAL_HEREINAFTER_RE, 6),
    ("this_agreement", LEGAL_THIS_AGREEMENT_RE, 4),
)

LEGAL_SUPPORT_CUES = (
    ("pursuant", LEGAL_PURSUANT_RE, 2),
    ("schedule", LEGAL_SCHEDULE_RE, 2),
    ("clause_section", LEGAL_CLAUSE_SECTION_RE, 1),
    ("governing_law", LEGAL_GOV_LAW_RE, 2),
    ("indemnity_liability", LEGAL_INDEMNITY_RE, 2),
    ("force_majeure", LEGAL_FORCE_MAJEURE_RE, 3),
    ("eu_cite", LEGAL_EU_CITE_RE, 3),
    ("act_year", LEGAL_ACT_YEAR_RE, 2),
)

@dataclass(frozen=True, slots=True)
class LegalConfig(DetectorConfig):
    enable_legal: bool = False
    sniff_threshold: int = 6

    # Future: defined-term related
    quoted_terms_only: bool = True
    term_context_window_chars: Optional[int] = 200
