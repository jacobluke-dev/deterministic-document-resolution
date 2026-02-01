import re
from typing import Optional

from plainera_unacronym.nlp.common.types import Span
from plainera_unacronym.nlp.extraction import ExtractionConfig
from plainera_unacronym.nlp.extraction.core.collect import initials_match
from plainera_unacronym.nlp.extraction.matchers.defs import (find_inline_longform_after_acr,
                                                             find_parenthetical_longform_before_acr,
                                                             find_parenthetical_longform_after_acr)


OptSpan = Optional[Span]

_POSSESSIVE_JOIN_RE = re.compile(r"\s*(?:['’]s\b)?\s*(?:[,;:—–-]\s*)?")
_QUOTE_CHARS = set("\"'“”‘’")
_TAIL_PUNCT = set(",;:—–-")


def _trim_span(seg: str, d0: int, d1: int) -> Span:
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
    # 1) quotes around acronym inside wrapper: ("PDF") / ('PDF') / (“PDF”)
    q_before = m.start("acr") - 1
    q_after = m.end("acr")
    has_quotes = (
        (0 <= q_before < len(seg) and seg[q_before] in _QUOTE_CHARS) or
        (0 <= q_after < len(seg) and seg[q_after] in _QUOTE_CHARS)
    )

    # 2) explicit tail punctuation after acronym: (PPE - ...), (PPE, ...), etc.
    tail_slice = seg[m.end("acr"): m.end()]
    has_tail = any(ch in _TAIL_PUNCT for ch in tail_slice)

    # 3) dotted acronym with terminal dot inside wrapper: (U.S.A.)
    post = m.end("acr")
    has_wrapper_dot = (post < len(seg) and seg[post] == ".")

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
    acr_end_local: int = None,
    m: re.Match[str] = None,
    cfg: ExtractionConfig,
) -> OptSpan:
    if kind == "def_after":
        assert acr_end_local is not None
        return _calc_def_span_def_after(acr_norm=acr_norm, seg=seg, acr_end_local=acr_end_local, cfg=cfg)

    if kind == "def_before":
        assert m is not None
        return _calc_def_span_def_before(acr_norm=acr_norm, seg=seg, m=m, cfg=cfg)

    # inline
    assert acr_end_local is not None
    return _calc_def_span_inline_after(acr_norm=acr_norm, seg=seg, acr_end_local=acr_end_local, cfg=cfg)


def resolve_def_span(strategy: str, *, seg: str, m: re.Match[str], acr_key: str, a1_local: int,
                     cfg: ExtractionConfig) -> OptSpan:
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
