import re
from plainera_unacronym.nlp.heuristics.shared import DottedMode


APOSTROPHE_VARIANTS = {
    "’": "'",  # U+2019
    "‘": "'",  # U+2018
    "ʼ": "'",  # U+02BC
    "′": "'",  # U+2032 (prime, often misused)
    "＇": "'",  # U+FF07 fullwidth
    "ʹ": "'",  # U+02B9 modifier letter
    "'": "'",  # U+0027
}

DASH_MAP = {"–": "-", "—": "-", "-": "-"}   # en/em/minus -> "-"

TRAILING_PUNCT = ",.;:!?)]}»”"
LEADING_BRACK  = "([«“["
CLOSING_BRACK  = ")]»”]"

STANDS_FOR_RE  = re.compile(r"\bstands\s+for\b", re.IGNORECASE)

BOUNDARY_TERMINATORS = ".!?…"            # includes unicode ellipsis
CLOSING_QUOTES_BRACKETS = "\")]}»”’"
BOUNDARY = ".!?\n\r\"'“”‘’([{"
TIME_RE  = re.compile(r"^(?:[01]?\d|2[0-3])(?::[0-5]\d)?$")  # 7, 10:30, 23:59

ALLOW_CHARS = "&/’'--–"

DOT_MODE : DottedMode = "strip"

PLURAL_SUFFIXES = ("s", "’s", "'s")
