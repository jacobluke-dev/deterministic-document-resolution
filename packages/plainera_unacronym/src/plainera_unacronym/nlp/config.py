import re

APOSTROPHE_VARIANTS = {"'": "’", "’": "’"}  # normalize to curly for keying

TRAILING_PUNCT = ",.;:!?)]}»”"
LEADING_BRACK  = "([«“["
CLOSING_BRACK  = ")]»”]"

STANDS_FOR_RE  = re.compile(r"\bstands\s+for\b", re.IGNORECASE)
