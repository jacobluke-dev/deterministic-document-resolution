import re
from dataclasses import dataclass
from typing import Optional

from document_resolution.nlp.common.types import AcronymDetectorConfig

STRUCT_SECTION_RE = re.compile(r"\bsection\s+\d+(?:\.\d+)*\b", re.IGNORECASE)
STRUCT_CLAUSE_RE = re.compile(r"\bclause\s+\d+(?:\.\d+)*\b", re.IGNORECASE)
STRUCT_SCHEDULE_RE = re.compile(r"\bschedule\s+[A-Z0-9IVX]+\b", re.IGNORECASE)
STRUCT_APPENDIX_RE = re.compile(r"\bappendix\s+[A-Z0-9IVX]+\b", re.IGNORECASE)
STRUCT_EXHIBIT_RE = re.compile(r"\bexhibit\s+[A-Z0-9IVX]+\b", re.IGNORECASE)
STRUCT_ANNEX_RE = re.compile(r"\bannex\s+[A-Z0-9IVX]+\b", re.IGNORECASE)
STRUCT_ARTICLE_RE = re.compile(r"\barticle\s+[A-Z0-9IVX]+\b", re.IGNORECASE)
STRUCT_PART_RE = re.compile(r"\bpart\s+[A-Z0-9IVX]+\b", re.IGNORECASE)
STRUCT_CHAPTER_RE = re.compile(r"\bchapter\s+\d+(?:\.\d+)*\b", re.IGNORECASE)

STRUCTURAL_STRONG_CUES = (
    ("schedule", STRUCT_SCHEDULE_RE, 4),
    ("appendix", STRUCT_APPENDIX_RE, 4),
    ("exhibit", STRUCT_EXHIBIT_RE, 4),
    ("annex", STRUCT_ANNEX_RE, 4),
)

STRUCTURAL_SUPPORT_CUES = (
    ("section", STRUCT_SECTION_RE, 2),
    ("clause", STRUCT_CLAUSE_RE, 2),
    ("article", STRUCT_ARTICLE_RE, 2),
    ("part", STRUCT_PART_RE, 2),
    ("chapter", STRUCT_CHAPTER_RE, 2),
)


@dataclass(frozen=True, slots=True)
class StructuralReferenceConfig(AcronymDetectorConfig):
    enable_structural_reference: bool = False
    sniff_threshold: int = 4
    sniff_cap_chars: Optional[int] = 80_000
