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
    quoted_term = r'(?P<term_q>"[A-Z][A-Za-z0-9&/\- ]{1,100}")'
    bare_term = r"(?P<term_b>[A-Z][A-Za-z0-9&/\-]+(?:\s+[A-Z][A-Za-z0-9&/\-]+){0,7})"

    quoted_means = re.compile(rf"{quoted_term}\s+means\b", re.IGNORECASE)
    quoted_shall_mean = re.compile(rf"{quoted_term}\s+shall\s+mean\b", re.IGNORECASE)

    bare_means = re.compile(rf"\b{bare_term}\s+means\b")
    bare_shall_mean = re.compile(rf"\b{bare_term}\s+shall\s+mean\b")

    quoted_occurrence = re.compile(r'"(?P<term>[A-Z][A-Za-z0-9&/\- ]{{1,100}})"')
    capitalised_occurrence = re.compile(
        r"\b(?P<term>[A-Z][A-Za-z0-9&/\-]+(?:\s+[A-Z][A-Za-z0-9&/\-]+){0,7})\b"
    )

    return DefinedTermPatterns(
        quoted_means=quoted_means,
        quoted_shall_mean=quoted_shall_mean,
        bare_means=bare_means,
        bare_shall_mean=bare_shall_mean,
        quoted_occurrence=quoted_occurrence,
        capitalised_occurrence=capitalised_occurrence,
    )
