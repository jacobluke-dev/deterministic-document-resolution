import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DefinedTermPatterns:
    quoted_means: re.Pattern[str]
    quoted_shall_mean: re.Pattern[str]
    bare_means: re.Pattern[str]
    bare_shall_mean: re.Pattern[str]
    parenthetical_alias: re.Pattern[str]
    quoted_occurrence: re.Pattern[str]
    capitalised_occurrence: re.Pattern[str]


def compile_defined_term_patterns() -> DefinedTermPatterns:
    hws = r"[ \t]+"

    quoted_term_char = r"A-Za-z0-9&/'’.\-()"
    bare_term_char = r"A-Za-z0-9&/'’\-()"

    bridge = r"(?:of|for|to|and|or|the|in|on|at|by|per)"

    quoted_head = rf"[A-Z][{quoted_term_char}]*"
    bare_head = rf"[A-Z][{bare_term_char}]*"

    quoted_tail = rf"(?:{hws}(?:{quoted_head}|{bridge}))*"
    bare_tail = rf"(?:{hws}(?:{bare_head}|{bridge}))*"

    quoted_term = rf'(?P<term_q>"{quoted_head}{quoted_tail}")'
    bare_term = rf"(?P<term_b>{bare_head}{bare_tail})"

    quoted_means = re.compile(rf"{quoted_term}{hws}means\b", re.IGNORECASE)
    quoted_shall_mean = re.compile(rf"{quoted_term}{hws}shall{hws}mean\b", re.IGNORECASE)

    bare_means = re.compile(rf"\b{bare_term}{hws}means\b")
    bare_shall_mean = re.compile(rf"\b{bare_term}{hws}shall{hws}mean\b")

    parenthetical_alias = re.compile(
        rf"\((?:the{hws})?(?P<term_q>\"{quoted_head}{quoted_tail}\")\)",
        re.IGNORECASE,
    )

    quoted_occurrence = re.compile(rf'"(?P<term>{quoted_head}{quoted_tail})"')
    capitalised_occurrence = re.compile(rf"\b(?P<term>{bare_head}{bare_tail})\b")

    return DefinedTermPatterns(
        quoted_means=quoted_means,
        quoted_shall_mean=quoted_shall_mean,
        bare_means=bare_means,
        bare_shall_mean=bare_shall_mean,
        parenthetical_alias=parenthetical_alias,
        quoted_occurrence=quoted_occurrence,
        capitalised_occurrence=capitalised_occurrence,
    )
