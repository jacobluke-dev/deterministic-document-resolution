import re
from typing import Literal

APOSTROPHE_VARIANTS = {
    "’": "'",  # U+2019
    "‘": "'",  # U+2018
    "ʼ": "'",  # U+02BC
    "′": "'",  # U+2032 (prime, often misused)
    "＇": "'",  # U+FF07 fullwidth
    "ʹ": "'",  # U+02B9 modifier letter
    "'": "'",  # U+0027
}

DASH_MAP = {"–": "-", "—": "-", "-": "-"}  # en/em/minus -> "-"

TRAILING_PUNCT_DEFAULT = ",.;:!?)]}»”"
LEADING_BRACK = "([«“["
CLOSING_BRACK = ")]»”]"

STANDS_FOR_RE = re.compile(r"\bstands\s+for\b", re.IGNORECASE)

BOUNDARY_TERMINATORS = ".!?…"  # includes unicode ellipsis
CLOSING_QUOTES_BRACKETS = '")]}»”’'
BOUNDARY = ".!?\n\r\"'“”‘’([{"
TIME_RE = re.compile(r"^(?:[01]?\d|2[0-3])(?::[0-5]\d)?$")  # 7, 10:30, 23:59
POST_SPAN_TOKEN_RE = re.compile(
    r"\s+|[A-Za-z]+|!|\S", re.ASCII
)  # tokens after `e`: whitespace, alpha word, '!', or any other single char

ALLOW_CHARS = "&/’'--–"
TOKEN_SEPARATORS = "-&/._"
EXCLAMS = ("!", "！", "‼")

DottedMode = Literal["strip", "preserve"]
DOT_MODE_DEFAULT: DottedMode = "strip"

PLURAL_SUFFIXES_DEFAULT = ("’s", "'s", "s")

CANON_TABLE_DEFAULT = {ord(k): ord(v) for k, v in {**APOSTROPHE_VARIANTS, **DASH_MAP}.items()}
