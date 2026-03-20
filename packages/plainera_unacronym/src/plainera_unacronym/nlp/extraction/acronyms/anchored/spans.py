from __future__ import annotations

import re
from typing import Optional

from plainera_unacronym.nlp.common.types import Span
from plainera_unacronym.nlp.extraction.acronyms.config import ExtractionConfig
from plainera_unacronym.nlp.extraction.acronyms.core.collect import initials_match
from plainera_unacronym.nlp.extraction.acronyms.matchers.defs import (
    find_inline_longform_after_acr,
    find_parenthetical_longform_after_acr,
    find_parenthetical_longform_before_acr,
)

OptSpan = Optional[Span]

_POSSESSIVE_JOIN_RE = re.compile(r"\s*(?:['’]s\b)?\s*(?:[,;:—–-]\s*)?")
_QUOTE_CHARS = set("\"'“”‘’")
_TAIL_PUNCT = set(",;:—–-")


def _trim_span(seg: str, d0: int, d1: int) -> Span:
    """Trim leading/trailing whitespace from a slice span.

    Adjusts (d0, d1) inward while the characters at the boundaries are
    whitespace (per `str.isspace()`), without touching internal whitespace.

    Args:
        seg: The string being spanned.
        d0: Start index (inclusive).
        d1: End index (exclusive).

    Returns:
        A tuple (new_d0, new_d1) such that `seg[new_d0:new_d1]` has no leading
        or trailing whitespace relative to the original `seg[d0:d1]` slice.
        If the slice is all-whitespace, returns (k, k) for some k in [d0, d1].
    """
    while d0 < d1 and seg[d0].isspace():
        d0 += 1
    while d1 > d0 and seg[d1 - 1].isspace():
        d1 -= 1
    return d0, d1


def _calc_def_span_def_before(
    *,
    acr_norm: str,
    seg: str,
    m: re.Match[str],
    cfg: ExtractionConfig,
) -> OptSpan:
    """Resolve a definition span for the forward-parenthetical form: `DEF (ACR...)`.

    Handles two cases:

    1) **Plain wrapper** `Long Form (ACR)`:
       Uses `find_parenthetical_longform_before_acr` on the matched snippet to
       reuse the “helper” logic and return its definition span.

    2) **Complex wrapper** where the acronym is decorated inside the wrapper:
       - quotes around acronym: `("ACR")`, `('ACR')`, `(“ACR”)`
       - trailing tail punctuation: `(ACR, ...)`, `(ACR - ...)`, etc.
       - dotted acronym with terminal dot inside wrapper: `(U.S.A.)`

       In these cases, bypasses the helper and returns the trimmed `m.group("def")`
       span **only if** `initials_match(acr_norm, phrase)` passes.

    Args:
        acr_norm: Normalised acronym key used for initials validation.
        seg: The local segment/window being searched.
        m: Regex match containing named groups `def` and `acr`.
        cfg: Extraction configuration passed through to helper matchers.

    Returns:
        (d0, d1) indices into `seg` for the selected definition, or None when no
        valid span can be resolved.
    """

    # 1) quotes around acronym inside wrapper: ("PDF") / ('PDF') / (“PDF”)
    q_before = m.start("acr") - 1
    q_after = m.end("acr")
    has_quotes = (0 <= q_before < len(seg) and seg[q_before] in _QUOTE_CHARS) or (
        0 <= q_after < len(seg) and seg[q_after] in _QUOTE_CHARS
    )

    # 2) explicit tail punctuation after acronym: (PPE - ...), (PPE, ...), etc.
    tail_slice = seg[m.end("acr") : m.end()]
    has_tail = any(ch in _TAIL_PUNCT for ch in tail_slice)

    # 3) dotted acronym with terminal dot inside wrapper: (U.S.A.)
    post = m.end("acr")
    has_wrapper_dot = post < len(seg) and seg[post] == "."

    # Complex wrapper: bypass helper, but require initials alignment (SLA safety)
    if has_quotes or has_tail or has_wrapper_dot:
        d0, d1 = _trim_span(seg, *m.span("def"))
        if d0 >= d1:
            return None
        phrase = seg[d0:d1]
        if not initials_match(acr_norm, phrase):
            return None
        return d0, d1

    # Plain case: "Long Form (ACR)" — keep helper behaviour
    snippet = seg[: m.end()]
    mm = find_parenthetical_longform_before_acr(snippet, acr_norm, cfg)
    if not mm:
        return None

    loc = mm[0]
    return loc.def_start, loc.def_end


def _calc_def_span_inline_after(
    *,
    acr_norm: str,
    seg: str,
    acr_end_local: int,
    cfg: ExtractionConfig,
) -> OptSpan:
    """Resolve the definition span for an inline-after pattern.

    This helper is used for patterns like:

        "ACR stands for Long Form"

    It slices the segment at `acr_end_local`, runs the inline-after matcher on the
    suffix, and (if a match is found) re-bases the matcher’s local definition span
    back into `seg` offsets.

    Args:
        acr_norm (str): Normalised acronym (typically uppercased) to match.
        seg (str): Local text segment being scanned.
        acr_end_local (int): End offset of the acronym within `seg` (exclusive).
        cfg (ExtractionConfig): Extraction configuration (uses `max_phrase_chars`).

    Returns:
        Span | None: `(def_start, def_end)` offsets into `seg` if a definition is
        found; otherwise `None`.
    """
    snippet = seg[acr_end_local:]
    mm = find_inline_longform_after_acr(
        snippet,
        cfg,
        acr=acr_norm,
        max_chars=cfg.max_phrase_chars * 2,
        require_initials_match=True,
    )
    if not mm:
        return None

    loc = mm[0]
    return acr_end_local + loc.def_start, acr_end_local + loc.def_end


def _calc_def_span_def_after(
    *,
    acr_norm: str,
    seg: str,
    acr_end_local: int,
    cfg: ExtractionConfig,
) -> OptSpan:
    """Resolve the definition span for an acronym-first parenthetical/bracket form.

    Handles shapes like:

        "ACR (Long Form)"
        "ACR’s (Long Form)"
        "ACR, (Long Form)"
        "ACR - (Long Form)"

    It slices `seg` at `acr_end_local`, optionally consumes a possessive/joiner
    (e.g. `'s`, commas, dashes) via `_POSSESSIVE_JOIN_RE`, then runs the
    acronym-after matcher on the remaining snippet. If a match is found, the
    matcher’s local definition span is re-based back into `seg` coordinates.

    Args:
        acr_norm (str): Normalised acronym (typically uppercased) to match.
        seg (str): Local text segment being scanned.
        acr_end_local (int): End offset of the acronym within `seg` (exclusive).
        cfg (ExtractionConfig): Extraction configuration passed to the matcher.

    Returns:
        Span | None: `(def_start, def_end)` offsets into `seg` if a definition is
        found; otherwise `None`.
    """
    snippet = seg[acr_end_local:]

    j = _POSSESSIVE_JOIN_RE.match(snippet)
    join_off = j.end() if j else 0
    snippet2 = snippet[join_off:]

    mm = find_parenthetical_longform_after_acr(
        snippet2,
        cfg,
        acr=acr_norm,
        require_initials_match=True,
    )
    if not mm:
        return None

    loc = mm[0]
    return (
        acr_end_local + join_off + loc.def_start,
        acr_end_local + join_off + loc.def_end,
    )


def _calc_def_span(
    kind: str,
    *,
    acr_norm: str,
    seg: str,
    acr_end_local: int | None = None,
    m: re.Match[str] | None = None,
    cfg: ExtractionConfig,
) -> OptSpan:
    """
    Compute the local definition span for an anchored extraction pattern.

    Selects the appropriate span-calculation strategy based on `kind` and derives
    a `(start, end)` character span relative to `seg`. Some strategies require
    additional context (e.g. the acronym end offset or the regex match object),
    which must be provided by the caller for the corresponding `kind`.

    Args:
        kind: Strategy identifier (e.g. "def_after", "def_before").
        acr_norm: Normalised acronym surface.
        seg: Text segment containing the match.
        acr_end_local: Local end offset of the acronym within `seg`; required
            when `kind == "def_after"`.
        m: Regex match object for the anchored pattern; required when
            `kind == "def_before"`.
        cfg: Extraction configuration used by the span calculators.

    Returns:
        An optional `(start, end)` span into `seg` representing the extracted
        definition region, or `None` if no valid span can be computed.
    """
    if kind == "def_after":
        assert acr_end_local is not None
        return _calc_def_span_def_after(acr_norm=acr_norm, seg=seg, acr_end_local=acr_end_local, cfg=cfg)

    if kind == "def_before":
        assert m is not None
        return _calc_def_span_def_before(acr_norm=acr_norm, seg=seg, m=m, cfg=cfg)

    # inline
    assert acr_end_local is not None
    return _calc_def_span_inline_after(acr_norm=acr_norm, seg=seg, acr_end_local=acr_end_local, cfg=cfg)


def resolve_def_span(
    strategy: str, *, seg: str, m: re.Match[str], acr_key: str, a1_local: int, cfg: ExtractionConfig
) -> OptSpan:
    """
    Resolve the definition span for an anchored pattern match.

    Given a pattern `strategy` and its regex match `m`, this function returns a
    `(start, end)` character span into `seg` identifying the definition region.
    Strategies either:
      - use the match group's span directly ("direct_def"), or
      - delegate to helper span calculators for more contextual selection
        (e.g. definition before/after the acronym, or inline cue forms).

    Args:
        strategy: Strategy identifier attached to a `PatternSpec` (e.g. "direct_def",
            "helper_def_after", "helper_def_before", "helper_inline_after").
        seg: Text segment containing the match.
        m: Regex match object for the anchored pattern (must include a "def" group
            for "direct_def").
        acr_key: Normalised acronym surface used by helper strategies.
        a1_local: Local end offset of the acronym within `seg` (used by "after"/"inline"
            helper strategies).
        cfg: Extraction configuration used by helper span calculators.

    Returns:
        An optional `(start, end)` span into `seg` representing the extracted
        definition region, or `None` if no valid span can be computed.
    """
    if strategy == "direct_def":
        d0, d1 = m.span("def")
        return None if d0 >= d1 else (d0, d1)
    if strategy == "helper_def_after":
        return _calc_def_span("def_after", acr_norm=acr_key, seg=seg, acr_end_local=a1_local, cfg=cfg)
    if strategy == "helper_def_before":
        return _calc_def_span("def_before", acr_norm=acr_key, seg=seg, m=m, cfg=cfg)
    if strategy == "helper_inline_after":
        return _calc_def_span("inline", acr_norm=acr_key, seg=seg, acr_end_local=a1_local, cfg=cfg)
    return None
