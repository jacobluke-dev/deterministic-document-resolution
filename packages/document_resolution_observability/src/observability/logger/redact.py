from typing import Any

from observability.config import SENSITIVE_KEYS, TOKEN_PATS


def _scrub_str(s: str) -> str:
    """Redact sensitive token-like substrings from a string.

    Args:
        s: Input string to scrub.

    Returns:
        The input string with any token-like substrings replaced by
        ``"[REDACTED]"``.

    """
    if s == "[REDACTED]":
        return s
    for pat in TOKEN_PATS:
        s = pat.sub("[REDACTED]", s)
    return s


def scrub(obj: Any) -> Any:
    """Recursively redact sensitive values in common JSON-like structures.

    The function is **non-destructive** (returns a new structure) and
    **idempotent** (re-scrubbing a result yields the same result).

    Args:
        obj: Any JSON-like structure (``dict``, ``list``, ``tuple``),
            or a scalar (``str``, ``int``, etc.).

    Returns:
        A new structure with sensitive fields redacted.

    """
    if isinstance(obj, dict):
        return {k: ("[REDACTED]" if k.lower() in SENSITIVE_KEYS else scrub(v)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [scrub(v) for v in obj]
        return type(obj)(t) if isinstance(obj, tuple) else t
    if isinstance(obj, str):
        return _scrub_str(obj)
    return obj
