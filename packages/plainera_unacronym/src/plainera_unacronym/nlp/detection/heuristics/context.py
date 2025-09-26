from typing import TYPE_CHECKING, FrozenSet, Protocol, cast, overload

from plainera_unacronym.nlp import DetectorConfig
from plainera_unacronym.nlp.common.constants import BOUNDARY, TIME_RE
from plainera_unacronym.nlp.common.shared import has_paren_definition
from plainera_unacronym.nlp.detection.heuristics.core import (
    has_stands_for_follow,
    in_brackets,
    next_word_lowercase,
    prev_token,
)
from plainera_unacronym.nlp.detection.heuristics.general import (
    at_sentence_boundary,
    is_all_caps_heading,
    is_all_caps_word,
    is_in_caps_interjection_context,
    is_in_caps_interjection_context_prev,
)


class HeuristicCfg(Protocol):
    """Structural subset of DetectorConfig used by context.py.

    Read-only properties so it matches a frozen/dataclass or @property-based config.
    """

    @property
    def allow_chars(self) -> FrozenSet[str]: ...
    @property
    def non_acronym_upper(self) -> FrozenSet[str]: ...
    @property
    def soft_blacklist(self) -> FrozenSet[str]: ...
    @property
    def enable_dotted(self) -> bool: ...


if TYPE_CHECKING:
    from plainera_unacronym.nlp.types import DetectorConfig

    CfgLike = HeuristicCfg | DetectorConfig

    def _assert_subset(x: DetectorConfig) -> DetectorConfig:
        return x
else:
    CfgLike = HeuristicCfg


def _in_definition_context(text: str, start: int, end: int) -> bool:
    inside, _ = in_brackets(text, start, end)
    return inside or has_paren_definition(text, end) or has_stands_for_follow(text, end)


def _drop_interjection(surface: str, text: str, s: int, e: int, cfg: CfgLike) -> bool:
    # general.py expects HeuristicCfg; cast once here

    hcfg = cast(HeuristicCfg, cfg)
    return is_in_caps_interjection_context(surface, text, s, e, hcfg) or is_in_caps_interjection_context_prev(
        surface, text, s, e, hcfg
    )


def _drop_all_caps_heading(surface: str, text: str, s: int, e: int, cfg: HeuristicCfg) -> bool:
    return is_all_caps_word(surface, cfg.allow_chars) and is_all_caps_heading(text, s, e)


def _is_sentence_start_i_am(text: str, start: int) -> bool:
    # … "I AM" at sentence start (allows leading spaces)
    i = start - 1
    while i >= 0 and text[i].isspace():
        i -= 1
    if i >= 0 and text[i] == "I":
        j = i - 1
        while j >= 0 and text[j].isspace():
            j -= 1
        return j < 0 or text[j] in BOUNDARY
    return False


def _token_specific_drop(tok: str, text: str, start: int, end: int) -> bool | None:
    if tok == "IT":
        return at_sentence_boundary(text, start) and next_word_lowercase(text, end)
    if tok == "AM":
        prev = prev_token(text, start)
        return bool(TIME_RE.match(prev)) or _is_sentence_start_i_am(text, start)
    return None


def _non_acronym_punct_or_lowercase_follow(text: str, end: int) -> bool:
    # After spaces, punctuation ,.!?;: → drop; or next word is lowercase → drop
    i, n = end, len(text)
    while i < n and text[i].isspace():
        i += 1
    if i < n and text[i] in ",.!?;:":
        return True
    return next_word_lowercase(text, end)


@overload
def blacklist_context_drop(surface: str, text: str, start: int, end: int, cfg: HeuristicCfg) -> bool: ...
@overload
def blacklist_context_drop(surface: str, text: str, start: int, end: int, cfg: "DetectorConfig") -> bool: ...


def blacklist_context_drop(surface: str, text: str, start: int, end: int, cfg: CfgLike) -> bool:
    """
    Decide whether to **drop** a candidate acronym based on local context.

    This applies a series of short-circuit heuristics to reject spans that are
    likely not acronyms (e.g., shouty interjections, headings, time tokens, or
    known non-acronym uppers followed by punctuation/lowercase). The checks are
    ordered from strongest “keep” signals to more general drop rules.

    The function returns **True** to drop/reject the candidate, **False** to keep it.

    Args:
        surface (str): The matched surface text (typically `text[start:end]`), e.g. "OK", "R&D", "IT".
        text (str): Full source text.
        start (int): Start offset (inclusive) of `surface` in `text`.
        end (int): End offset (exclusive) of `surface` in `text`.
        cfg (HeuristicCfg): Detection config. Uses:
            - `allow_chars` (for separator checks via other helpers),
            - `non_acronym_upper` (known uppercase tokens like "OK", "PM"),
            - optional `blacklist` (extra tokens to always consider for dropping).

    Returns:
        bool: `True` if the candidate should be dropped; `False` to keep.

    Decision order:
        0. **Never drop** when near explicit definitions:
           - Inside brackets/parentheses (`in_brackets`), or
           - Followed by parenthetical definition (`has_paren_definition`), or
           - “stands for …” pattern to the right (`has_stands_for_follow`).
        1. **Drop** shouty ALL-CAPS interjections:
           - `is_in_caps_interjection_context` (or the previous-token variant).
        1b. **Drop** ALL-CAPS headings:
           - If `is_all_caps_word(surface, cfg.allow_chars)` and `is_all_caps_heading(...)`.
        2. Token-specific polysemes:
           - `"IT"` → drop when at a sentence boundary **and** the next word is lowercase.
           - `"AM"` → drop when preceded by a time token (e.g., `"9 AM"`), or sentence-start `"I AM …"`.
        3. If `surface` is **not** in `cfg.blacklist` **and** not in `cfg.non_acronym_upper` → keep (`False`).
        4. Known non-acronym uppers:
           - **Drop** if followed by punctuation `, . ! ? ; :` (after spaces), or if the next word is lowercase.
        5. Generic fallback:
           - **Drop** when at a sentence boundary **and** the next word is lowercase.

    Notes:
        - Offsets are `[start, end)` (end-exclusive).
        - Helper predicates used: `in_brackets`, `has_paren_definition`, `has_stands_for_follow`,
          `is_in_caps_interjection_context`, `is_in_caps_interjection_context_prev`,
          `is_all_caps_word`, `is_all_caps_heading`, `at_sentence_boundary`,
          `next_word_lowercase`, `prev_token`, and `TIME_RE`.
        - The rules are conservative and ordered to minimize false drops of genuine acronyms.

    Examples:
        - `"OK,"` at the start of a clause → dropped (known non-acronym upper followed by punctuation).
        - `"R&D"` in running text → kept (not blacklisted; separators handled elsewhere).
        - `"IT"` at sentence start followed by lowercase word → dropped.
        - `"9 AM"` → dropped (time pattern).
        - `"(ABC) stands for …"` → kept (definition context).
    """
    tok = surface

    # 0) Definition/expansion context → never drop
    if _in_definition_context(text, start, end):
        return False

    # 1) Shouty interjection
    if _drop_interjection(surface, text, start, end, cfg):
        return True

    # 1b) ALL-CAPS heading
    if _drop_all_caps_heading(surface, text, start, end, cast(HeuristicCfg, cfg)):
        return True

    # 2) Token-specific polysemes
    decision = _token_specific_drop(tok, text, start, end)
    if decision is not None:
        return decision

    # 3) From here, only consider explicit lists
    if tok not in getattr(cfg, "blacklist", frozenset()) and tok not in cfg.non_acronym_upper:
        return False

    # 4) Known non-acronym uppers: punctuation/lowercase continuation
    if tok in cfg.non_acronym_upper and _non_acronym_punct_or_lowercase_follow(text, end):
        return True

    # 5) Generic fallback
    return at_sentence_boundary(text, start) and next_word_lowercase(text, end)
