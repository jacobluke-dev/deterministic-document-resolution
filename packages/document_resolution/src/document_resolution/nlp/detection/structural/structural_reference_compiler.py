from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class StructuralReferencePatterns:
    """Container for compiled regex patterns used by structural-reference detection.
    """

    schedule_reference: re.Pattern[str]
    exhibit_reference: re.Pattern[str]
    annex_reference: re.Pattern[str]
    appendix_reference: re.Pattern[str]
    section_reference: re.Pattern[str]
    clause_reference: re.Pattern[str]
    article_reference: re.Pattern[str]


def compile_structural_reference_patterns() -> StructuralReferencePatterns:
    """Compile and return the regex patterns used by the structural-reference detector.

    Supported forms include:
        - ``Schedule A`` / ``Schedule 1``
        - ``Exhibit B``
        - ``Annex 1``
        - ``Appendix C``
        - ``Section 4`` / ``Section 4.2``
        - ``Clause 7`` / ``Clause 7.3``
        - ``Article III`` / ``Article 2``

    Returns:
        A ``StructuralReferencePatterns`` instance containing compiled regular
        expression objects for each supported structural-reference kind.
    """

    hws = r"[ \t]+"  # horizontal whitespace

    alpha_label = r"[A-Z]"
    numeric_label = r"\d+"
    decimal_label = r"\d+(?:\.\d+)*"
    roman_label = r"[IVXLCDM]+"
    appendix_label = r"(?:[A-Z](?:\.\d+)*|\d+)"

    schedule_reference = re.compile(
        rf"\b(?P<kind>Schedule){hws}(?P<label>{alpha_label}|{numeric_label})\b",
        re.IGNORECASE,
    )
    exhibit_reference = re.compile(
        rf"\b(?P<kind>Exhibit){hws}(?P<label>{alpha_label}|{numeric_label})\b",
        re.IGNORECASE,
    )
    annex_reference = re.compile(
        rf"\b(?P<kind>Annex){hws}(?P<label>{alpha_label}|{numeric_label})\b",
        re.IGNORECASE,
    )
    appendix_reference = re.compile(
        rf"\b(?P<kind>Appendix){hws}(?P<label>{appendix_label})\b",
        re.IGNORECASE,
    )
    section_reference = re.compile(
        rf"\b(?P<kind>Section){hws}(?P<label>{decimal_label})\b",
        re.IGNORECASE,
    )
    clause_reference = re.compile(
        rf"\b(?P<kind>Clause){hws}(?P<label>{decimal_label})\b",
        re.IGNORECASE,
    )
    article_reference = re.compile(
        rf"\b(?P<kind>Article){hws}(?P<label>{roman_label}|{numeric_label})\b",
        re.IGNORECASE,
    )

    return StructuralReferencePatterns(
        schedule_reference=schedule_reference,
        exhibit_reference=exhibit_reference,
        annex_reference=annex_reference,
        appendix_reference=appendix_reference,
        section_reference=section_reference,
        clause_reference=clause_reference,
        article_reference=article_reference,
    )
