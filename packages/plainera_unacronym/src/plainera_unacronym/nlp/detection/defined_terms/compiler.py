import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DefinedTermPatterns:
    quoted_means: re.Pattern
    quoted_shall_mean: re.Pattern
    bare_means: re.Pattern
    bare_shall_mean: re.Pattern
    quoted_occurrence: re.Pattern
    capitalised_occurrence: re.Pattern


def compile_defined_term_patterns() -> DefinedTermPatterns:
    term_char = r"A-Za-z0-9&/'’.\-()"
    bridge = r"(?:of|for|to|and|or|the|in|on|at|by|per)"
    head = rf"[A-Z][{term_char}]*"
    tail = rf"(?:\s+(?:{head}|{bridge}))*"

    quoted_term = rf'(?P<term_q>"{head}{tail}")'
    bare_term = rf"(?P<term_b>{head}{tail})"

    quoted_means = re.compile(rf"{quoted_term}\s+means\b", re.IGNORECASE)
    quoted_shall_mean = re.compile(rf"{quoted_term}\s+shall\s+mean\b", re.IGNORECASE)

    bare_means = re.compile(rf"\b{bare_term}\s+means\b")
    bare_shall_mean = re.compile(rf"\b{bare_term}\s+shall\s+mean\b")

    quoted_occurrence = re.compile(rf'"(?P<term>{head}{tail})"')
    capitalised_occurrence = re.compile(rf"\b(?P<term>{head}{tail})\b")

    return DefinedTermPatterns(
        quoted_means=quoted_means,
        quoted_shall_mean=quoted_shall_mean,
        bare_means=bare_means,
        bare_shall_mean=bare_shall_mean,
        quoted_occurrence=quoted_occurrence,
        capitalised_occurrence=capitalised_occurrence,
    )
