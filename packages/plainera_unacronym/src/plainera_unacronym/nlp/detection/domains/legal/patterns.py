import re
from functools import lru_cache


@lru_cache(maxsize=1)
def legal_pattern() -> re.Pattern[str]:
    # Mixed alpha + roman numerals: MiFID II, CRR III, Basel IV, etc.
    roman = r"(?:I{1,4}|V|VI|VII|VIII|IX|X)"
    named_regime = rf"(?:MiFID|CRR|Basel|Solvency|MAR|EMIR)\s+{roman}"

    # EU regulation/directive citations: 2016/679, 2014/65/EU etc.
    eu_cite = r"(?:\b(?:Regulation|Directive)\s*(?:\(\s*EU\s*\))?\s*\d{4}/\d{2,4}(?:/EU)?\b)"

    # UK/US Act style: Data Protection Act 2018
    act_year = r"(?:\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4}\s+Act\s+\d{4}\b)"

    # Standards commonly referenced in legal/compliance docs
    iso = r"(?:\bISO\s+\d{4,5}\b)"

    return re.compile(rf"(?P<legal>{named_regime}|{eu_cite}|{act_year}|{iso})")
