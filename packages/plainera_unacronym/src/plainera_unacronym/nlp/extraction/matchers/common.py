import re
from typing import Optional

from plainera_unacronym.nlp.common.constants_regex import PUNCT_TRIM

_ASCII_CAMEL_RE = re.compile(
    r"[A-Z]+(?=[A-Z][a-z0-9])"  # e.g., 'XML' in 'XMLHttp'
    r"|[A-Z]?[a-z]+[0-9]*"  # word with optional trailing digits, e.g., 'v1'
    r"|[0-9]+"  # standalone digits
)

LEXICAL_SPLITS = {
    # Networking / protocols / web
    "websocket": ("Web", "Socket"),  # WS (less common), but appears a lot
    "middleware": ("Middle", "Ware"),  # MW (internal docs)
    "firmware": ("Firm", "Ware"),  # FW
    "hardware": ("Hard", "Ware"),  # HW
    "software": ("Soft", "Ware"),  # SW (can collide with "switch", but as a split it's fine)

    # Identity / auth / accounts
    "hostname": ("Host", "Name"),  # HN
    "password": ("Pass", "Word"),  # PW (super common in docs)

    # Storage / data
    "database": ("Data", "Base"),  # DB (historically ugly, but extremely common)
    # Languages
    "typescript": ("Type", "Script"),  # TS (collides with timestamp)
    "powershell": ("Power", "Shell"),  # PS (collides heavily)

    # Platforms / tools
    "bitbucket": ("Bit", "Bucket"),  # BB
    "gitlab": ("Git", "Lab"),  # GL
    "github": ("Git", "Hub"),  # GH

    "postgresql": ("Postgres", "SQL"),  # PG/PSQL alignment
    "mysql": ("My", "SQL"),  # MySQL is already Camel-ish, but tokenisers often keep as one
    "mssql": ("MS", "SQL"),  # MS SQL / MSSQL

    "newline": ("New", "Line"),  # NL
    "filepath": ("File", "Path"),  # FP
    "filename": ("File", "Name"),  # FN
    "checksum": ("Check", "Sum"),  # CS
    "hypertext": ("Hyper", "text"),
}


def match_from(letters: list[str], acronym_list: list[str], start: int) -> Optional[tuple[int, list[int]]]:
    """
    Greedily align A as a subsequence of letters starting at index `start`.
    Returns (end_index_exclusive_in_letters, matched_letter_positions) or None.
    """
    li, ai = start, 0
    used = []
    while li < len(letters) and ai < len(acronym_list):
        if letters[li] == acronym_list[ai]:
            used.append(li)
            ai += 1
        li += 1
    if ai == len(acronym_list):
        return li, used
    return None

def is_mixed_case_acronym(acr: str) -> bool:
    letters = [c for c in acr if c.isalpha()]
    return any(c.islower() for c in letters) and any(c.isupper() for c in letters)


def initials_seq(tokens: list[str], *, expand_allcaps: bool = False) -> tuple[list[str], list[int]]:
    """
    Build a sequence of initials (letters+digits) from tokens.
    owners[k] = token index that produced letters[k].

    Unicode-aware: picks the first character in each part where ch.isalpha() or ch.isdigit().

    Special-case: ALL-CAPS alphabetic tokens (e.g., "RNA") contribute *all* letters
    so mixed-case acronyms like "mRNA" can align to phrases like "messenger RNA".
    """
    letters, owners = [], []
    for ti, tok in enumerate(tokens):
        tok_clean = tok.strip(PUNCT_TRIM)

        if expand_allcaps and tok_clean.isalpha() and tok_clean.isupper() and len(tok_clean) > 1:
            for ch in tok_clean:
                letters.append(ch.upper())
                owners.append(ti)
            continue

        for part in split_compound(tok_clean):
            for ch in part:
                if ch.isalpha() or ch.isdigit():
                    letters.append(ch.upper())
                    owners.append(ti)
                    break
    return letters, owners


def split_compound(token: str) -> list[str]:
    """Split hyphen/slash/dot/& and (ASCII) CamelCase into parts.

    Rules:
    - Non-ASCII pieces (e.g., 'Ångström') are kept intact (no Camel split).
    - ASCII pieces with both letters and digits that START or END with a digit
      are kept intact as a single part (e.g., '3D', 'v1').
    - Otherwise, ASCII CamelCase is split using _ASCII_CAMEL_RE.
    """
    pieces = re.split(r"[\-\/\.\&]", token)
    out: list[str] = []
    for p in pieces:
        if not p:
            continue
        if not re.fullmatch(r"[A-Za-z0-9]+", p):
            # Contains non-ASCII or other chars -> keep whole piece
            out.append(p)
            continue

        has_alpha = bool(re.search(r"[A-Za-z]", p))
        has_digit = bool(re.search(r"[0-9]", p))

        # ---- SPECIAL-CASE LEXICAL COMPOUNDS ----
        low = p.lower()
        if low in LEXICAL_SPLITS:
            out.extend(LEXICAL_SPLITS[low])
            continue

        # Keep leading-digit+letters or trailing-digit combos intact: '3D', 'v1', 'HTTP2'
        if has_alpha and has_digit and (p[0].isdigit() or p[-1].isdigit()):
            out.append(p)
            continue

        parts = _ASCII_CAMEL_RE.findall(p)
        out.extend(parts if parts else [p])

    return out
