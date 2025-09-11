from typing import Literal


def has_paren_definition(text: str, end: int, max_chars: int = 80) -> bool:
    i, n = end, len(text)
    while i < n and text[i].isspace(): i += 1
    if i < n and text[i] == "(":
        j, alpha = i + 1, 0
        while j < n and (j - i) <= max_chars and text[j] != ")":
            if text[j].isalpha(): alpha += 1
            j += 1
        return j < n and alpha >= 5
    return False


DottedMode = Literal["strip", "preserve", "both"]
