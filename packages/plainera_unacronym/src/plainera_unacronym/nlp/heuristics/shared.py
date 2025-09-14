from typing import Literal


def has_paren_definition(text: str, end: int, max_chars: int = 80) -> bool:
    """Return whether a parenthetical definition follows immediately after a token.

        Starting at ``end`` (the character index directly after a candidate token),
        this checks for optional whitespace, then a ``'('`` and scans up to
        ``max_chars`` characters **inside** the parentheses looking for a closing
        ``')'``. It counts alphabetic characters within the parentheses and returns
        ``True`` only if a closing parenthesis is found within the limit and at
        least 5 letters are present (letters are determined via ``str.isalpha()``,
        so Unicode letters like Greek count).

        Args:
          text: The full source text being analyzed.
          end: The index in ``text`` immediately after the token to test.
          max_chars: Maximum number of characters to scan **inside** the parentheses
            before giving up. The closing ``')'`` must appear within this window.

        Returns:
          True if a parenthetical definition is detected within the limit and
          contains at least 5 letters; otherwise False.

        Examples:
          >>> s = "GPU (Graphics Processing Unit) is common."
          >>> has_paren_definition(s, s.index("GPU") + 3)
          True
          >>> s = "CPU (org) used here."
          >>> has_paren_definition(s, s.index("CPU") + 3)
          False
          >>> s = "ATP (αβγδε) binding"
          >>> has_paren_definition(s, s.index("ATP") + 3)
          True
        """
    i, n = end, len(text)
    # skip whitespace
    while i < n and text[i].isspace():
        i += 1
    # must have '(' next
    if not (i < n and text[i] == "("):
        return False

    j, alpha = i + 1, 0
    limit = i + 1 + max_chars  # scan at most max_chars chars *inside* the parens

    while j < n and j <= limit and text[j] != ")":
        if text[j].isascii() and text[j].isalpha():
            alpha += 1
        j += 1

    # valid only if we hit a closing ')' within the limit
    return (j < n and text[j] == ")") and (alpha >= 5)


DottedMode = Literal["strip", "preserve"]
