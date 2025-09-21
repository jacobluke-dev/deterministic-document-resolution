# extraction/helpers_patterns.py (or wherever you keep extraction helpers)
from dataclasses import dataclass
import re
from typing import List

from plainera_unacronym.nlp.common.types import ExtractedDefinition


@dataclass
class LocalDefMatch:
    def_start: int   # start index within the snippet you passed
    def_end: int     # end index (exclusive) within the snippet
    definition: str  # raw definition text as captured

# --- small label normalizer used before turning a capture into a "sense" label ---
def tighten_label(s: str) -> str:
    """
    Keep definition crisp for senses:
      - use your normalize_definition (NFKC, collapse ws, strip trailing punct)
      - drop leading determiners like 'the', 'a', 'an'
    """
    from plainera_unacronym.nlp.common.shared import normalize_definition
    s = normalize_definition(s)
    s = re.sub(r'^(?:the|a|an)\s+', '', s, flags=re.IGNORECASE)
    return s

# -------------- pattern finders ----------------

def find_longform_after_acr(snippet: str, acr: str, cfg) -> List[LocalDefMatch]:
    """
    Look for:  ACR ( Long form ... )  immediately after the ACR.
    Caller should slice `snippet` so its index 0 == position right after ACR,
    or pass the whole snippet and also pass the relative 'acr_end' substring.
    For simplicity here we assume the caller already sliced to start at acr_end.
    """
    max_chars = getattr(cfg, "max_phrase_chars", 80)
    # Immediately optional spaces, then a parenthetical
    pat = re.compile(
        r"""\A              # start of the given substring
            \s*             # optional spaces
            \(
              (?P<def>[^()]{1,%d})
            \)
        """ % max_chars,
        re.VERBOSE,
    )
    m = pat.match(snippet)
    if not m:
        return []
    raw = m.group("def").strip()
    # quick sanity: require a few letters so we don't capture junk
    if sum(ch.isalpha() for ch in raw) < 3:
        return []
    return [LocalDefMatch(def_start=m.start("def"), def_end=m.end("def"), definition=raw)]


def find_longform_before_acr(snippet: str, acr: str, cfg) -> List[LocalDefMatch]:
    """
    Look for:  Long form ... ( ACR )  ending right before the ACR.
    Caller should slice `snippet` so its end == position at ACR start,
    or pass full snippet and we anchor to the end.
    Here we anchor to the end of the given substring.
    """
    max_chars = getattr(cfg, "max_phrase_chars", 80)
    # We want "...(ACR)" flush at the end of the substring
    # Capture the long form immediately before the "(ACR)".
    acr_escaped = re.escape(acr)
    pat = re.compile(
        rf"""
        (?P<def>[^\(\)]{{1,{max_chars}}})   # long form with no parentheses
        \s*                                 # optional spaces
        \(\s*{acr_escaped}\s*\)            # literal (ACR)
        \s*$                                # then end of substring
        """,
        re.VERBOSE,
    )
    m = pat.search(snippet)
    if not m:
        return []
    raw = m.group("def").strip()
    if sum(ch.isalpha() for ch in raw) < 3:
        return []
    return [LocalDefMatch(def_start=m.start("def"), def_end=m.end("def"), definition=raw)]


def dedupe_defs(defs: list[ExtractedDefinition]) -> list[ExtractedDefinition]:
    """Dedupe on (acronym.upper(), tighten_label(definition)). Keep earliest span, merge support later."""
    seen = set()
    out = []
    for d in defs:
        key = (d.acronym.upper(), tighten_label(d.definition))
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out
