from typing import Any

from observability.config import SENSITIVE_KEYS, TOKEN_PATS


def _scrub_str(s: str) -> str:
    """Redact sensitive token-like substrings from a string.

    This function removes bearer/basic tokens and similar key/value patterns
    based on a small set of regexes. It is **idempotent** (re-running it on an
    already-scrubbed string yields the same string) and will not alter the
    special placeholder ``"[REDACTED]"``.

    Args:
        s: Input string to scrub.

    Returns:
        The input string with any token-like substrings replaced by
        ``"[REDACTED]"``.

    Examples:
        >>> _scrub_str("Bearer abcdefghijkLMNOP")
        '[REDACTED]'
        >>> _scrub_str("note=ok Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ== end")
        'note=ok [REDACTED] end'
        >>> _scrub_str("[REDACTED]")
        '[REDACTED]'
    """
    if s == "[REDACTED]":
        return s
    for pat in TOKEN_PATS:
        s = pat.sub("[REDACTED]", s)
    return s


def scrub(obj: Any) -> Any:
    """Recursively redact sensitive values in common JSON-like structures.

    Behavior:
      * **Dicts** – For keys whose lowercase form is in ``SENSITIVE_KEYS``,
        replace the value with ``"[REDACTED]"``. Keys are matched
        case-insensitively but **original casing is preserved** in the output.
      * **Lists/Tuples** – Recurse into each element. Tuples are preserved as
        tuples; lists remain lists.
      * **Strings** – Redact token-like substrings via :func:`_scrub_str`.
      * **Other types** – Returned unchanged.

    The function is **non-destructive** (returns a new structure) and
    **idempotent** (re-scrubbing a result yields the same result).

    Args:
        obj: Any JSON-like structure (``dict``, ``list``, ``tuple``),
            or a scalar (``str``, ``int``, etc.).

    Returns:
        A new structure with sensitive fields redacted.

    Examples:
        >>> scrub({"Authorization": "Bearer abcdefghijk", "ok": 1})
        {'Authorization': '[REDACTED]', 'ok': 1}
        >>> scrub(["Bearer abcdefghijk", {"password": "p"}])
        ['[REDACTED]', {'password': '[REDACTED]'}]
    """
    if isinstance(obj, dict):
        return {k: ("[REDACTED]" if k.lower() in SENSITIVE_KEYS else scrub(v)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [scrub(v) for v in obj]
        return type(obj)(t) if isinstance(obj, tuple) else t
    if isinstance(obj, str):
        return _scrub_str(obj)
    return obj
