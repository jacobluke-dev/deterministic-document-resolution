import re

APOSTROPHE_VARIANTS = {"'": "’", "’": "’"}  # normalize to curly for keying

TRAILING_PUNCT = ",.;:!?)]}»”"
LEADING_BRACK  = "([«“["
CLOSING_BRACK  = ")]»”]"

STANDS_FOR_RE  = re.compile(r"\bstands\s+for\b", re.IGNORECASE)

BOUNDARY = ".!?\n\r\"'“”‘’([{"
TIME_RE  = re.compile(r"^(?:[01]?\d|2[0-3])(?::[0-5]\d)?$")  # 7, 10:30, 23:59

allow_chars = "&/’'--–"
