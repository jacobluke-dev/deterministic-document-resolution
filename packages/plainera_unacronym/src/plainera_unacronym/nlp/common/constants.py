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

# Bridges are the words we're willing to keep for readability inside the span
BRIDGES_DEFAULT: frozenset[str] = frozenset(DEFAULT_WORDS_SHARED | {"&"})

LINKERS = {"of","and","for","to","in","on","with","the","per","by","via","as","at","from","vs","&"}
_LINKERS_RE = "(?:" + "|".join(sorted(re.escape(w) for w in LINKERS)) + ")"
_TITLE = r"[A-Z][\w’'\u2011-]*"  # TitleCase/ALL-CAPS, allow unicode apostrophes + NB hyphen
_DASH = r"[–—-]"                 # en/em/ascii dash

# TitleCase token: allow letters/digits/underscore, Unicode apostrophes, NB hyphen (U+2011), ASCII hyphen
_TITLECASE_TOKEN = r"[A-Z][\w’'\u2011-]*"

# TitleCase run: TitleCase ( (space+TitleCase) | (space+(linker|dash)+space+TitleCase) )*
# TITLECASE_RUN_RE = re.compile(
#     rf"({_TITLE}(?:\s+(?:{_TITLE}|(?:{_LINKERS_RE}|{_DASH})\s+{_TITLE}))*)",
#     flags=re.UNICODE,
# )

# Last TitleCase/ALL-CAPS run at the **end** of the string,
# allowing common linkers (incl. '&') or a dash between TitleCase tokens.
TITLECASE_RUN_RE = re.compile(
    rf"(?:^|[\s,])"                          # start or whitespace/comma before the run
    rf"("                                     # capture the whole tail run
      rf"{_TITLE}"                            # TitleCase/ALL-CAPS token
      rf"(?:\s+(?:"
        rf"{_TITLE}"                          # another TitleCase/ALL-CAPS token
        rf"|(?:{_LINKERS_RE}|{_DASH})\s+{_TITLE}"  # linker/dash + TitleCase token
      rf"))*"                                 # repeat joins
    rf")\s*$",                                # must be at end (tail)
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
