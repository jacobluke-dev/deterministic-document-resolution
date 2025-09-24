import re
from typing import Optional
from plainera_unacronym.nlp.common.types import ExtractedDefinition
from ..common.shared import tighten_definition_span, normalize_definition, tighten_label


class LocalDefMatch:
    def __init__(self, def_start: int, def_end: int, definition: str):
        self.def_start = def_start
        self.def_end = def_end
        self.definition = definition


def _has_letters(s: str) -> bool:
    """True if the string contains any Unicode letter.

    Args:
      s (str): String to check.

    Returns:
      bool: True if any character in ``s`` satisfies ``str.isalpha()``; else False.
    """
    return any(ch.isalpha() for ch in s)


def _initials_match(acr: str, phrase: str) -> bool:
    """Check if an acronym fits the phrase's initials as an ordered subsequence.

        Builds an uppercase string of initials from the phrase by taking the first
        character of each word **only if** that character is alphabetic. Then checks
        whether the alphabetic characters of ``acr`` (ignoring any non-letters in
        ``acr``) appear in order within those initials.

        This is case-insensitive for matching and does not require contiguity—only
        order. Words that begin with non-letters (e.g., ``"3M"``, ``"7-Document"``)
        do not contribute an initial.

        Args:
          acr (str): The Acronym to test.
          phrase (str): Candidate long-form phrase used to derive initials.

        Returns:
          bool: True if the acronym's letters appear in order within the phrase initials;
          otherwise False.

        """
    initials = "".join(w[0].upper() for w in phrase.split() if w and w[0].isalpha())
    j = 0
    for ch in acr:
        if ch.isalpha():
            j = initials.find(ch, j) + 1
            if j == 0:
                return False
    return True


def find_longform_after_acr(
    snippet: str,
    cfg,
    acr: Optional[str] = None,
    require_acrostic: bool = True,
) -> list[LocalDefMatch]:
    """
    Look for:  ACR ( Long form ... )  immediately after the ACR.
    Caller should slice `snippet` so its index 0 == position right after ACR,
    or pass the whole snippet and also pass the relative 'acr_end' substring.
    For simplicity here we assume the caller already sliced to start at acr_end.
    """
    max_chars = getattr(cfg, "max_phrase_chars", 80)
    pat = re.compile(rf"\A\s*\((?P<def>[^()]{{1,{max_chars}}})\)", re.VERBOSE)

    m = pat.match(snippet)
    if not m:
        return []

    raw = m.group("def")
    if not _has_letters(raw):
        return []

    # normalize & tighten like the main pipeline
    norm = normalize_definition(tighten_definition_span(raw))
    if not norm:
        return []

    # Optional acrostic guard to reduce junk like "(see below)"
    if acr and require_acrostic:
        if not _initials_match(acr.upper(), norm):
            return []

    return [LocalDefMatch(def_start=m.start("def"), def_end=m.end("def"), definition=norm)]


def find_longform_before_acr(snippet: str, acr: str, cfg) -> list[LocalDefMatch]:
    """
    Look for:  Long form ... ( ACR )  ending right before the ACR.
    Caller should slice `snippet` so its end == position at ACR start,
    or pass full snippet, and we anchor to the end.
    Here we anchor to the end of the given substring.
    """
    max_chars = getattr(cfg, "max_phrase_chars", 80)
    acr_escaped = re.escape(acr)
    pat = re.compile(
        rf"(?P<def>[^\(\)]{{1,{max_chars}}})\s*\(\s*{acr_escaped}\s*\)\s*$",
        re.VERBOSE,
    )
    m = pat.search(snippet)
    if not m:
        return []

    raw = m.group("def")
    if not _has_letters(raw):
        return []

    norm = normalize_definition(tighten_definition_span(raw))
    if not norm:
        return []

    # Before-ACR is already anchored on the correct (ACR)
    return [LocalDefMatch(def_start=m.start("def"), def_end=m.end("def"), definition=norm)]


def dedupe_defs(defs: list[ExtractedDefinition]) -> list[ExtractedDefinition]:
    """Dedupe on (acronym.upper(), tighten_label(definition)). Keep the earliest span, merge support later."""
    seen = set()
    out = []
    for d in defs:
        key = (d.acronym.upper(), tighten_label(d.definition))
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out
