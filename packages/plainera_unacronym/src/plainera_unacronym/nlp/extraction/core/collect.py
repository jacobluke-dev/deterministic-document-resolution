import re
from typing import Iterator

from plainera_unacronym.nlp.common.shared import normalize_definition, collapse_ws, strip_trailing_punct_str
from plainera_unacronym.nlp.extraction.anchored.normalise import tighten_definition_span
from plainera_unacronym.nlp.extraction.matchers.helper_patterns import has_letters
from plainera_unacronym.nlp.extraction.matchers.tighten import tighten_label_by_acronym

from plainera_unacronym.nlp.common.types import ExtractedDefinition
from plainera_unacronym.nlp.extraction.config import ExtractionConfig

def _two_words(s: str) -> bool:
    return len([w for w in s.split() if re.search(r"[A-Za-z]", w)]) >= 2


def _parenthetical_allowed(cfg: ExtractionConfig, definition: str, acronym: str) -> bool:
    allows: list[Callable[[str, str], bool]] = getattr(cfg, "_parenthetical_allows", [])  # type: ignore[attr-defined]
    return all(fn(definition, acronym) for fn in allows) if allows else True

def initials_match(acr: str, phrase: str) -> bool:
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
            j = initials.find(ch.upper(), j) + 1
            if j == 0:
                return False
    return True


def _collapses_to_acronym(defn: str, acr: str) -> bool:
    d = normalize_definition(defn).strip()
    if not d:
        return True
    # Common cases: "SLA", "(SLA)", "SLA."
    d = strip_trailing_punct_str(d).strip()
    return d.upper() == acr.upper()



def collect_matches(
    text: str,
    pat: re.Pattern[str],
    *,
    cfg: ExtractionConfig,
    plan: "ExtractionPlan",
    base_conf: float,
    is_parenthetical: bool,
    seen: set[tuple[int, int, int, int]],
    start: int,
    end: int,
) -> Iterator[ExtractedDefinition]:
    for m in pat.finditer(text, start, end):
        acr_raw, def_raw = m.group("acr"), m.group("def")
        if not acr_raw or not def_raw:
            continue

        acronym = strip_trailing_punct_str(acr_raw.strip()).upper()
        if not (cfg.min_acr_len <= len(acronym) <= cfg.max_acr_len):
            continue

        # Inline defs should not span lines; skip overreach matches
        if not is_parenthetical and ("\n" in def_raw or "\r" in def_raw):
            continue

        raw_trim = def_raw.strip()
        if not raw_trim:
            continue

        # --- RAW length gate for INLINE (gate, don't truncate) ---
        # If the regex captured a large chunk (search_cap), we still enforce cfg.max_phrase_chars here.
        if not is_parenthetical:
            raw_gate = collapse_ws(raw_trim)
            if len(raw_gate) > cfg.max_phrase_chars:
                continue

        original_def = def_raw

        # Normalise/tighten (display)
        definition = normalize_definition(tighten_definition_span(raw_trim))
        if not definition:
            continue
        if len(definition) > cfg.max_phrase_chars:
            continue
        if not has_letters(definition):
            continue
        if cfg.require_two_words and not _two_words(definition):
            continue

        # Apply label tightening ONCE and treat that as final candidate
        final_def = tighten_label_by_acronym(
            definition,
            acronym.upper(),
            stopwords=set(cfg.stop),
            bridges=set(cfg.bridges),
        )
        final_def = normalize_definition(final_def)
        if not final_def:
            continue
        if len(final_def) > cfg.max_phrase_chars:
            continue
        if cfg.require_two_words and not _two_words(final_def):
            continue

        if is_parenthetical:
            # 1) prevent "definition == acronym"
            if strip_trailing_punct_str(final_def).strip().upper() == acronym:
                continue

            # 2) require initials plausibility for parentheticals
            if not initials_match(acronym, final_def):
                continue

        # Parenthetical-specific gating / “3M” preservation
        if is_parenthetical:
            # If the first token starts non-alpha (e.g., 3M, 10GbE) and got dropped, reattach it.
            m_numtok = re.match(r"^(\S+)", raw_trim)
            if m_numtok:
                first_tok = m_numtok.group(1)
                if re.match(r"^[^A-Za-z]", first_tok) and not final_def.startswith(first_tok):
                    final_def = f"{first_tok} {final_def}".strip()
                    final_def = normalize_definition(final_def)
                    if not final_def or len(final_def) > cfg.max_phrase_chars:
                        continue

            # Config-driven allows
            if not _parenthetical_allowed(cfg, final_def, acronym):
                continue
            # Plan-driven allows (plugins)
            if plan.parenthetical_allows and not all(fn(final_def, acronym) for fn in plan.parenthetical_allows):
                continue

        a0, a1 = m.span("acr")
        d0, d1 = m.span("def")
        key = (a0, a1, d0, d1)
        if key in seen:
            continue
        seen.add(key)

        conf = min(base_conf + (0.03 if initials_match(acronym, final_def) else 0.0), 0.99)

        yield ExtractedDefinition(
            acronym=acronym,
            definition=final_def,
            source="in_text",
            confidence=conf,
            acr_start=a0,
            acr_end=a1,
            def_start=d0,
            def_end=d1,
            original_definition=original_def,
        )
