import re
from typing import Final, Literal


# -----------------------------
# Canonical normalisation maps
# -----------------------------

APOSTROPHE_VARIANTS: Final[dict[str, str]] = {
    "’": "'",  # U+2019
    "‘": "'",  # U+2018
    "ʼ": "'",  # U+02BC
    "′": "'",  # U+2032 (prime, often misused)
    "＇": "'",  # U+FF07 fullwidth
    "ʹ": "'",  # U+02B9 modifier letter
    "'": "'",  # U+0027
}

# Dash folding (keep as exported name; used by CANON_TABLE_DEFAULT)
DASH_MAP: Final[dict[str, str]] = {"–": "-", "—": "-", "-": "-"}

# Translation table for fast canonicalisation
CANON_TABLE_DEFAULT: Final[dict[int, int]] = {
    ord(k): ord(v) for k, v in {**APOSTROPHE_VARIANTS, **DASH_MAP}.items()
}


# -----------------------------
# Articles / connectors
# -----------------------------

ARTICLE: Final[re.Pattern[str]] = re.compile(r"^(?:the|an|a)\s+", flags=re.IGNORECASE)

LEADING_CONNECTORS: Final[re.Pattern[str]] = re.compile(
    r"^(?:while|whereas|and|or|but|that|which|who|as|for|to)\b[\s,:-]*",
    flags=re.IGNORECASE,
)


# -----------------------------
# “Function words” / stopwords / bridges
# Single source of truth to prevent drift
# -----------------------------

DEFAULT_WORDS_SHARED: Final[frozenset[str]] = frozenset(
    {
        # English-ish function words
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
        "via",
        "vs",
        "&",
    }
)

NAMED_STOPWORDS: Final[frozenset[str]] = frozenset(
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

# Canonical set (everything derives from this)
WORDS_FUNCTION: Final[frozenset[str]] = frozenset(DEFAULT_WORDS_SHARED | NAMED_STOPWORDS)

DEFAULT_STOPWORDS: Final[frozenset[str]] = WORDS_FUNCTION

# Bridges are words we keep inside spans for readability
BRIDGES_DEFAULT: Final[frozenset[str]] = WORDS_FUNCTION  # includes "&" already via DEFAULT_WORDS_SHARED


# Legacy exports (keep them to avoid breaking imports)
LINKERS: Final[set[str]] = set(WORDS_FUNCTION)  # historically a set
_LINKERS_RE: Final[str] = "(?:" + "|".join(sorted(re.escape(w) for w in WORDS_FUNCTION)) + ")"


# -----------------------------
# Boundaries / punctuation
# -----------------------------

# Dashes (ascii + en/em)
_DASH: Final[str] = r"[–—-—–-]"

# punctuation / clause boundaries used for splitting
BOUNDARY_RE: Final[re.Pattern[str]] = re.compile(rf"[\.!?;:,—–-]\s+")

BOUNDARY_TERMINATORS: Final[str] = ".!?…"  # includes unicode ellipsis
CLOSING_QUOTES_BRACKETS: Final[str] = '")]}»”’'

# A general “boundary char” set used by some scanners
BOUNDARY: Final[str] = ".!?\n\r\"'“”‘’([{"

TRAILING_PUNCT_CHARS: Final[str] = ",.;:!?)]}»”"
TRAILING_PUNCT_DEFAULT: Final[re.Pattern[str]] = re.compile(rf"[{re.escape(TRAILING_PUNCT_CHARS)}\s]+$")

LEADING_BRACK: Final[str] = "([«“["
CLOSING_BRACK: Final[str] = ")]»”]"

EXCLAMS: Final[tuple[str, ...]] = ("!", "！", "‼")


# -----------------------------
# Common regexes used across extraction
# -----------------------------

STANDS_FOR_RE: Final[re.Pattern[str]] = re.compile(r"\bstands\s+for\b", re.IGNORECASE)

TIME_RE: Final[re.Pattern[str]] = re.compile(r"^(?:[01]?\d|2[0-3])(?::[0-5]\d)?$")  # 7, 10:30, 23:59

POST_SPAN_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"\s+|[A-Za-z]+|!|\S", re.ASCII
)  # tokens after `e`: whitespace, alpha word, '!', or any other single char


# -----------------------------
# Acronym key / token parsing
# -----------------------------

ALLOW_CHARS: Final[str] = "&/’'--–"
TOKEN_SEPARATORS: Final[str] = "-&/._"

DottedMode = Literal["strip", "preserve"]
DOT_MODE_DEFAULT: Final[DottedMode] = "strip"

PLURAL_SUFFIXES_DEFAULT: Final[tuple[str, ...]] = ("’s", "'s", "s")

DEFAULT_TWO_LETTER_BOOST: Final[float] = 0.75


# -----------------------------
# Titlecase run detection (tighten_definition_span)
# -----------------------------

# Strict TitleCase / ALLCAPS token
_TITLE: Final[str] = r"[A-Z][\w’'\u2011-]*"

# Allow lower-case hyphenated tokens like "sign-on", "end-to-end" (but NOT plain "print")
_LOWER_HYPHEN: Final[str] = r"[a-z][a-z0-9’'\u2011-]*[\-\u2011][a-z0-9’'\u2011-]+"

# A token that can appear inside a "definition-ish" title run
_TITLELIKE: Final[str] = rf"(?:{_TITLE}|{_LOWER_HYPHEN})"

TITLECASE_RUN_RE: Final[re.Pattern[str]] = re.compile(
    rf"(?:^|[\s,])"
    rf"("
    rf"{_TITLELIKE}"
    rf"(?:\s+(?:"
    rf"{_TITLELIKE}"
    rf"|{_LINKERS_RE}\s+{_TITLELIKE}"
    rf"|{_DASH}\s+{_TITLELIKE}"
    rf"))*"
    rf")\s*$",
    flags=re.UNICODE,
)

Q = r"""["'“”‘’]"""
QUOTE = rf"(?:\s*{Q}\s*)?"

INLINE_CUE_FRAGMENTS: Final[tuple[str, ...]] = (
    r"short\s+for",
    r"stands?\s+for",
    r"is\s+(?:an\s+)?acronym\s+for",
    r"abbreviated\s+as",
)
