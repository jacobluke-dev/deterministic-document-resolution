import re
from typing import Optional

from document_resolution.nlp.common.constants_regex import PUNCT_TRIM
from document_resolution.nlp.common.shared import has_letter
from document_resolution.nlp.extraction.acronyms.core.normalise import has_digit

_ASCII_CAMEL_RE = re.compile(
    r"[A-Z]+(?=[A-Z][a-z0-9])"  # e.g., 'XML' in 'XMLHttp'
    r"|[A-Z]?[a-z]+[0-9]*"  # word with optional trailing digits, e.g., 'v1'
    r"|[0-9]+"  # standalone digits
    r"|[A-Z]+(?![a-z0-9])"  # trailing caps, e.g. X in TeX, AI in OpenAI
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


def should_preserve_alnum_token(token: str) -> bool:
    """Return True if an ASCII alphanumeric token should be kept intact.

    This is used by token splitting logic to avoid fragmenting tokens that are
    commonly treated as single units in technical text (e.g. "2FA", "HTTP2",
    "RFC7231", "B2B").

    The token must be ASCII alphanumeric and contain both letters and digits.
    It is preserved when:
      - It starts with a digit (e.g. "2FA", "7Zip"), or
      - It ends with a digit (e.g. "HTTP2", "v1", "x86"), or
      - It is all-uppercase (e.g. "B2B", "RFC7231", "SHA256", "H264").

    Args:
        token (str): Candidate token (already separator-split) to evaluate.

    Returns:
        bool: True if the token should be preserved as a single part; otherwise False.
    """
    if not token or not token.isalnum():
        return False

    if not (has_letter(token) and has_digit(token)):
        return False

    if token[0].isdigit() or token[-1].isdigit():
        return True

    # Key addition: ALLCAPS+digits like B2B (digits in the middle)
    return token.upper() == token


def match_from(letters: list[str], acronym_list: list[str], start: int) -> Optional[tuple[int, list[int]]]:
    """Greedily align an acronym as an ordered subsequence of `letters`, starting at `start`.

    Scans `letters` from index `start` onward and tries to match `acronym_list`
    in order (no backtracking). When a match is found, records the letter index
    used and advances to the next acronym character. If all acronym characters
    are matched, returns the scan end position and the indices that were used.

    Args:
        letters (list[str]): Stream of candidate letters (typically already uppercased).
        acronym_list (list[str]): Acronym characters to match in order.
        start (int): Index into `letters` to begin scanning from.

    Returns:
        Optional[tuple[int, list[int]]]:
            If successful, returns a tuple `(end_idx, used_positions)` where:
              - `end_idx` is the index in `letters` where the scan stopped
                (exclusive; i.e., the next position you would continue scanning from),
              - `used_positions` is the list of indices in `letters` that matched
                each character of `acronym_list` in order.
            Returns None if the acronym cannot be fully matched.

    Notes:
        - This is a greedy subsequence matcher: it finds the earliest possible
          completion given the starting point, but does not guarantee a globally
          optimal match if multiple alignments exist.
        - Caller is responsible for any normalisation (e.g., uppercasing, filtering
          to alnum) before passing inputs.

    Examples:
        >>> match_from(list("ABCD"), list("AC"), 0)
        (3, [0, 2])
        >>> match_from(list("ABCD"), list("DA"), 0) is None
        True
        >>> match_from(list("AAB"), list("AB"), 1)
        (3, [1, 2])
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
    """Return True if `acr` contains both uppercase and lowercase letters.

    Non-letter characters (digits, punctuation, symbols) are ignored for the
    purposes of the check.

    Args:
        acr (str): Acronym candidate to inspect.

    Returns:
        bool: True if the alphabetic characters in `acr` include at least one
        lowercase and at least one uppercase letter; otherwise False.
    """
    letters = [c for c in acr if c.isalpha()]
    return any(c.islower() for c in letters) and any(c.isupper() for c in letters)


def initials_seq(tokens: list[str], *, expand_allcaps: bool = False) -> tuple[list[str], list[int]]:
    """Build an initials stream (letters/digits) from a list of tokens.

    Produces:
      - `letters`: the extracted initials (uppercased), one per token-part.
      - `owners`: parallel list mapping each `letters[i]` back to the originating
        token index in `tokens`.

    Each token is first trimmed using `PUNCT_TRIM`, then split into sub-parts via
    `split_compound()` (e.g., hyphens, slashes, CamelCase). For each part, the
    first alphanumeric character contributes an initial.

    If `expand_allcaps` is True, ALL-CAPS alphabetic tokens of length > 1
    contribute *all* their letters (e.g., "RNA" -> "R","N","A"). This enables
    alignment of mixed-case acronyms such as "mRNA" to phrases like
    "messenger RNA".

    Args:
        tokens (list[str]): Token strings to derive initials from.
        expand_allcaps (bool): Whether to expand ALL-CAPS alphabetic tokens into
            multiple initials instead of taking a single initial.

    Returns:
        tuple[list[str], list[int]]: A `(letters, owners)` pair where `letters`
        are uppercased initials (letters/digits) and `owners[i]` is the token
        index that produced `letters[i]`.
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
    """Split a compound token into sub-parts for initials extraction and matching.

    This function performs *lightweight*, deterministic token segmentation using:
      1) hard separators: hyphen (`-`), slash (`/`), dot (`.`), ampersand (`&`)
      2) ASCII-only CamelCase / alnum chunking via `_ASCII_CAMEL_RE`
      3) optional lexical overrides via `LEXICAL_SPLITS`
      4) an alphanumeric preservation rule via `should_preserve_alnum_token()`

    Behaviour rules:
      - If a piece contains non-ASCII letters/symbols (i.e. not strictly `[A-Za-z0-9]`),
        it is kept intact (no CamelCase splitting).
      - If `LEXICAL_SPLITS` contains the lowercase form of a piece (e.g. "postgresql"),
        the configured split is used verbatim.
      - If `should_preserve_alnum_token(piece)` is True (e.g. for acronym-ish alnum like
        "2FA" or "HTTP2"), the piece is kept intact.
      - Otherwise, the piece is split using `_ASCII_CAMEL_RE` into parts such as
        "XML" + "Http" + "Request", or "Foo" + "Bar".

    Args:
        token (str): Input token to split (may include separators/CamelCase/alnum).

    Returns:
        list[str]: Ordered list of sub-parts derived from the token. Empty parts are
        not returned.
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

        # ---- SPECIAL-CASE LEXICAL COMPOUNDS ----
        low = p.lower()
        if low in LEXICAL_SPLITS:
            out.extend(LEXICAL_SPLITS[low])
            continue

        # ---- digit handling ----
        if should_preserve_alnum_token(p):
            out.append(p)
            continue

        parts = _ASCII_CAMEL_RE.findall(p)
        out.extend(parts if parts else [p])

    return out
