import re
from typing import Final, Literal

APOSTROPHE_VARIANTS = {
    "’": "'",  # U+2019
    "‘": "'",  # U+2018
    "ʼ": "'",  # U+02BC
    "′": "'",  # U+2032 (prime, often misused)
    "＇": "'",  # U+FF07 fullwidth
    "ʹ": "'",  # U+02B9 modifier letter
    "'": "'",  # U+0027
}

ARTICLE = re.compile(r"^(?:the|an|a)\s+", flags=re.IGNORECASE)

# punctuation / clause boundaries
BOUNDARY_RE = re.compile(r"[\.!?;:,—–-]\s+")

DEFAULT_TWO_LETTER_BOOST: Final[float] = 0.75

DEFAULT_WORDS_SHARED: frozenset[str] = frozenset(
    {
        # English-ish function words—expand as needed
        "of",
        "and",
        "the",
        "for",
        "to",
        "in",
        "on",
        "with",
        "a",
        "an",
        "at",
        "by",
        "from",
        "as",
        "per",
    }
)

NAMED_STOPWORDS: frozenset[str] = frozenset(
    {
        # Common non-English determiners/prepositions for names
        "de",
        "la",
        "le",
        "du",
        "des",
        "del",
        "da",
        "di",
        "von",
        "und",
    }
)

DEFAULT_STOPWORDS: frozenset[str] = frozenset(DEFAULT_WORDS_SHARED | NAMED_STOPWORDS)

# Bridges are the words you're willing to keep for readability inside the span
BRIDGES_DEFAULT: frozenset[str] = DEFAULT_WORDS_SHARED


# last TitleCase/UPPER run (optionally joined by common linkers)
TITLECASE_TAIL_RE = re.compile(
    r"(?:^|[\s,])([A-Z][\w’'-]*(?:\s+(?:[A-Z][\w’'-]*|of|and|for|to|in|on|with|the|&)){0,12})\s*$",
    flags=re.UNICODE,
)

LEADING_CONNECTORS = re.compile(
    r"^(?:while|whereas|and|or|but|that|which|who|as|for|to)\b[\s,:-]*",
    flags=re.IGNORECASE,
)

DASH_MAP = {"–": "-", "—": "-", "-": "-"}  # en/em/minus -> "-"

TRAILING_PUNCT_CHARS = ",.;:!?)]}»”"
TRAILING_PUNCT_DEFAULT = re.compile(rf"[{re.escape(TRAILING_PUNCT_CHARS)}\s]+$")
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
