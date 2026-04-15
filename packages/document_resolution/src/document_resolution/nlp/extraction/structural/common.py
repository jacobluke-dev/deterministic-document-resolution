import re

_ROMAN_STRICT_RE = re.compile(r"^(M{0,3})(CM|CD|D?C{0,3})" r"(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")


def is_strict_roman_numeral(value: str) -> bool:
    """Return whether a value is a well-formed Roman numeral.

    Validates the input against a strict Roman numeral pattern covering the
    conventional subtractive forms up to 3999, for example ``III``, ``IV``,
    ``IX``, ``XL``, ``XC``, ``CD``, and ``CM``.

    Args:
        value: Candidate Roman numeral text to validate.

    Returns:
        ``True`` if ``value`` is a well-formed Roman numeral; otherwise
        ``False``.
    """
    if not value:
        return False
    return _ROMAN_STRICT_RE.fullmatch(value.upper()) is not None


def roman_to_int(value: str) -> int:
    """Convert a well-formed Roman numeral string to an integer.

    The input is first validated using ``_is_strict_roman_numeral``. Conversion
    then proceeds right-to-left using standard Roman numeral subtraction rules.

    Args:
        value: Roman numeral text to convert, for example ``"III"``,
            ``"IV"``, or ``"MCMXCIV"``.

    Returns:
        Integer value of the Roman numeral.

    Raises:
        ValueError: If ``value`` is not a well-formed Roman numeral.
    """
    if not is_strict_roman_numeral(value):
        raise ValueError(f"Invalid Roman numeral: {value}")

    roman_map = {
        "I": 1,
        "V": 5,
        "X": 10,
        "L": 50,
        "C": 100,
        "D": 500,
        "M": 1000,
    }

    total = 0
    prev = 0
    for ch in reversed(value.upper()):
        curr = roman_map[ch]
        if curr < prev:
            total -= curr
        else:
            total += curr
            prev = curr
    return total
