from typing import TYPE_CHECKING, FrozenSet, Protocol, cast, overload

from plainera_unacronym.nlp.common.constants_regex import BOUNDARY, TIME_RE
from plainera_unacronym.nlp.common.shared import has_paren_definition
from plainera_unacronym.nlp.common.types import DetectorConfig
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
    """
    Structural subset of DetectorConfig required by context-based heuristics.

    This protocol lets context heuristics accept either the full DetectorConfig or a
    lightweight config implementation, while remaining type-safe and read-only.

    Attributes:
        allow_chars (FrozenSet[str]): Allowed internal separator characters.
        non_acronym_upper (FrozenSet[str]): Uppercase tokens treated as non-acronyms (e.g. OK, PM).
        soft_blacklist (FrozenSet[str]): Short, locale-aware blacklist of common words in caps.
        enable_dotted (bool): Whether dotted initialisms (e.g. U.S.A) are enabled.
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
    from plainera_unacronym.nlp.common.types import DetectorConfig

    CfgLike = HeuristicCfg | DetectorConfig

    def _assert_subset(x: DetectorConfig) -> DetectorConfig:
        return x
else:
    CfgLike = HeuristicCfg


def _in_definition_context(text: str, start: int, end: int) -> bool:
    """
    Detect “definition context” signals around a candidate span.

    Treats a token as definition-backed if it is inside brackets, followed by a
    parenthetical definition, or followed by a “stands for …” cue.

    Args:
        text (str): Full source text.
        start (int): Start offset (inclusive) of the candidate span.
        end (int): End offset (exclusive) of the candidate span.

    Returns:
        bool: True if the span is likely part of an explicit definition/expansion.
    """
    inside, _ = in_brackets(text, start, end)
    return inside or has_paren_definition(text, end) or has_stands_for_follow(text, end)


def _drop_interjection(surface: str, text: str, s: int, e: int, cfg: CfgLike) -> bool:
    """
    Drop candidates that appear in ALL-CAPS interjection contexts.

    Delegates to general heuristics that detect “shouty” tokens used as discourse markers
    (e.g., OK!, NO!, YES!) either at the current span or immediately before it.

    Args:
        surface (str): Candidate surface text.
        text (str): Full source text.
        s (int): Start offset (inclusive) of the candidate span.
        e (int): End offset (exclusive) of the candidate span.
        cfg (CfgLike): Config implementing the HeuristicCfg subset.

    Returns:
        bool: True if the token should be dropped as an interjection; else False.
    """
    # general.py expects HeuristicCfg; cast once here
    hcfg = cast(HeuristicCfg, cfg)
    return is_in_caps_interjection_context(surface, text, s, e, hcfg) or is_in_caps_interjection_context_prev(
        surface, text, s, e, hcfg
    )


def _drop_all_caps_heading(surface: str, text: str, s: int, e: int, cfg: HeuristicCfg) -> bool:
    """
    Drop candidates that are part of an ALL-CAPS heading.

    Uses a strict ALL-CAPS token predicate plus a heading-context predicate to suppress
    section headings that tend to produce noisy uppercase “matches”.

    Args:
        surface (str): Candidate surface text.
        text (str): Full source text.
        s (int): Start offset (inclusive) of the candidate span.
        e (int): End offset (exclusive) of the candidate span.
        cfg (HeuristicCfg): Config subset (uses allow_chars for ALL-CAPS checks).

    Returns:
        bool: True if the token should be dropped as a heading artifact; else False.
    """
    return is_all_caps_word(surface, cfg.allow_chars) and is_all_caps_heading(text, s, e)


def effective_blacklist(cfg: DetectorConfig) -> frozenset[str]:
    """
    Returns the complete black list system and user / organisational
    defined list.

    Args:
        cfg (DetectorConfig): Config implementing the HeuristicCfg subset.

    returns:
        frozenset[str]: A complete black list system and user / organisational defined list

    """
    return cfg.blacklist | cfg.user_org_blacklist


def _is_sentence_start_i_am(text: str, start: int) -> bool:
    """
    Detect the sentence-start pattern “I AM …” where AM is not an acronym.

    Walks left from `start` across whitespace and checks whether the previous non-space
    character is 'I' and that it is at document start or preceded by a boundary char.

    Args:
        text (str): Full source text.
        start (int): Start offset (inclusive) of the candidate span.

    Returns:
        bool: True if the token is an “AM” in the phrase “I AM” at a sentence start.
    """
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
    """
    Apply token-specific disambiguation rules for high-frequency polysemes.

    Currently special-cases tokens that often appear in normal prose:
      - "IT" at sentence boundary followed by lowercase word.
      - "AM" following a time token (e.g. “9 AM”) or in sentence-start “I AM …”.

    Args:
        tok (str): Candidate token (surface).
        text (str): Full source text.
        start (int): Start offset (inclusive).
        end (int): End offset (exclusive).

    Returns:
        bool | None: True to drop, False to keep, or None if no special-case applies.
    """
    if tok == "IT":
        return at_sentence_boundary(text, start) and next_word_lowercase(text, end)
    if tok == "AM":
        prev = prev_token(text, start)
        return bool(TIME_RE.match(prev)) or _is_sentence_start_i_am(text, start)
    return None


def _non_acronym_punct_or_lowercase_follow(text: str, end: int) -> bool:
    """
    Determine whether a non-acronym token is followed by punctuation or lowercase flow.

    Skips whitespace then checks for immediate clause punctuation (",.!?;:") or whether
    the next lexical word begins lowercase, both of which suggest discourse usage.

    Args:
        text (str): Full source text.
        end (int): End offset (exclusive) of the candidate span.

    Returns:
        bool: True if the follow-on context suggests “not an acronym”.
    """
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
    Decide whether to drop a candidate acronym based on local context and config lists.

    This is the context “driver” that combines several fast, short-circuit heuristics:
    definition-context immunity, interjection and heading suppression, token-specific
    polyseme rules, explicit blacklist/non-acronym lists, and a final sentence-boundary
    fallback. It returns True to drop and False to keep.

    Args:
        surface (str): Candidate surface text (typically `text[start:end]`).
        text (str): Full source text.
        start (int): Start offset (inclusive) of the candidate in `text`.
        end (int): End offset (exclusive) of the candidate in `text`.
        cfg (CfgLike): Config implementing the HeuristicCfg subset; may also include:
            - `blacklist` (optional FrozenSet[str]): tokens treated as context-droppable.

    Returns:
        bool: True if the candidate should be dropped; False if it should be kept.

    Decision order:
        0) Keep if in explicit definition context (brackets/paren def/“stands for”).
        1) Drop ALL-CAPS interjections (current or previous token context).
        2) Drop ALL-CAPS headings.
        3) Apply token-specific rules ("IT", "AM") and return if applicable.
        4) If not in cfg.blacklist and not in cfg.non_acronym_upper → keep.
        5) If in cfg.non_acronym_upper and followed by punct/lowercase → drop.
        6) Fallback: drop at sentence boundary when next word is lowercase.

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
