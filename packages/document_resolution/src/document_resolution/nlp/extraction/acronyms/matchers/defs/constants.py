from dataclasses import dataclass


class LocalDefMatch:
    def __init__(self, def_start: int, def_end: int, definition: str, raw: str | None = None):
        self.def_start = def_start
        self.def_end = def_end
        self.definition = definition
        self.raw = raw


@dataclass(frozen=True, slots=True)
class InitialsStream:
    """
    `letters` is a list of scan-order initials (uppercased),
    `owners[i]` is the token index that produced `letters[i]`, and
    `is_stop[i]` is the stopword status of the owning token.
    """

    letters: list[str]
    owners: list[int]
    is_stop: list[bool]


@dataclass(frozen=True, slots=True)
class AlignmentHit:
    """
    `used_letter_pos`: indices into `stream.letters` used by the match,
    `hit_tokens`: set of token indices that contributed initials,
    `tok_left`/`tok_right`: inclusive token-span bounds covering `hit_tokens`.
    """

    used_letter_pos: list[int]
    hit_tokens: set[int]
    tok_left: int
    tok_right: int

