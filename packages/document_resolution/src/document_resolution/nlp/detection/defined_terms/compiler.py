import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DefinedTermPatterns:
    """Container for compiled regex patterns used by defined-term detection.

    Each field stores a compiled regular expression for one supported drafting or
    occurrence pattern. These patterns are compiled once and reused by the
    detector to avoid repeated regex construction during scanning.

    Attributes:
        quoted_means: Matches quoted term introductions using ``means``, for
            example ``"Effective Date" means ...``.
        quoted_shall_mean: Matches quoted term introductions using
            ``shall mean``, for example ``"Services" shall mean ...``.
        bare_means: Matches unquoted capitalised term introductions using
            ``means``, for example ``Change of Control means ...``.
        bare_shall_mean: Matches unquoted capitalised term introductions using
            ``shall mean``, for example ``Confidential Information shall mean ...``.
        parenthetical_alias: Matches parenthetical alias definitions, for example
            ``(the "Agreement")`` or ``("Supplier")``.
        quoted_occurrence: Matches later quoted occurrences of a defined term, for
            example ``"Services"``.
        capitalised_occurrence: Matches later unquoted capitalised occurrences that
            may resolve to a known defined term, for example
            ``Change of Control``.
    """

    quoted_means: re.Pattern[str]
    quoted_shall_mean: re.Pattern[str]
    bare_means: re.Pattern[str]
    bare_shall_mean: re.Pattern[str]
    parenthetical_alias: re.Pattern[str]
    quoted_occurrence: re.Pattern[str]
    capitalised_occurrence: re.Pattern[str]


def compile_defined_term_patterns() -> DefinedTermPatterns:
    """Compile and return the regex patterns used by the defined-term detector.

    The compiled patterns cover a bounded set of supported drafting forms,
    including quoted introductions, selected bare capitalised introductions,
    parenthetical aliases, quoted occurrences, and broader capitalised occurrence
    runs.

    Args:
        None

    Returns:
        A ``DefinedTermPatterns`` instance containing compiled regular expression
        objects for supported introduction and occurrence patterns.

    Raises:
        re.error: If any regex expression is invalid at compile time.
    """
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
