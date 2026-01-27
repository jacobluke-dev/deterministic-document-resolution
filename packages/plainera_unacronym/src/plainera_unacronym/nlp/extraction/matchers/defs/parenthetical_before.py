import re

from plainera_unacronym.nlp.common.constants_regex import DEFAULT_STOPWORDS, BRIDGES_DEFAULT, QUOTE
from plainera_unacronym.nlp.common.shared import (has_letters,
                                                  strip_trailing_punct_str,
                                                  collapse_ws,
                                                  normalize_definition)
from plainera_unacronym.nlp.extraction.matchers.common import is_mixed_case_acronym
from plainera_unacronym.nlp.extraction.matchers.defs.common import (LocalDefMatch,
                                                                    first_alnum_char_upper,
                                                                    build_initials_stream,
                                                                    align_acronym_to_initials,
                                                                    expand_numeric_leading_window, build_kept_phrase)
from plainera_unacronym.nlp.extraction.matchers.numeric_matcher import consume_left_numeric_designator


def find_parenthetical_longform_before_acr(snippet: str, acr: str, cfg) -> list[LocalDefMatch]:
    """
    Find:  Long Form ... (ACR)   anchored at the end of `snippet`.

    Strategy (RIGHT→LEFT):
      1) Capture the text immediately before '(ACR)' with a look-ahead.
      2) Tokenize LTR (to get stable character spans), but *match* RTL.
      3) Split tokens into parts (hyphen/slash/dot/& and CamelCase).
         Build a per-part initials sequence RTL.
      4) Match ACR letters (ignoring non-alnum), with constraints:
         - UPPERCASE letter → must land on a non-stopword token
         - lowercase letter → must land on a stopword token
      5) The token window is [leftmost contributing token .. last token].
         Expand the window to include adjacent numeric-leading tokens.
      6) Keep matched tokens + bridges + numeric-leading tokens (for readability).
      7) Return tight character spans over the original `snippet` and a normalized phrase.
    """
    max_chars = getattr(cfg, "max_phrase_chars", 80)
    stop = getattr(cfg, "stopwords", DEFAULT_STOPWORDS)
    bridges = getattr(cfg, "bridges", BRIDGES_DEFAULT)

    acr_esc = re.escape(acr)

    tail = r"(?:\s*[,;:]\s*[^)]{0,120})?"  # cap the tail to stay sane

    m = re.search(
        rf"(?P<pre>[^\(\)]{{1,{max_chars}}})\s*"
        rf"(?=\(\s*{QUOTE}{acr_esc}{QUOTE}{tail}\s*\)\s*$)",
        snippet,
    )

    if not m:
        return []

    pre = m.group("pre").rstrip()
    if not pre or not has_letters(pre):
        return []

    # Raw length guard (before any tightening): enforce the configured limit strictly.
    if len(pre) > max_chars:
        return []

    # 1) Tokenize LTR by whitespace to get stable spans
    tokens: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for t in re.finditer(r"\S+", pre):
        tokens.append(t.group(0))
        starts.append(t.start())
        ends.append(t.end())

    if not tokens:
        return []

    acr_starts_with_digit = acr and acr[0].isdigit()

    stream = build_initials_stream(
        tokens,
        stopwords=stop,
        scan="rtl",
        expand_allcaps_tokens=False,
        split_compounds=True,
        treat_acronym_tokens_as_multi_letter=True,
    )

    hit = align_acronym_to_initials(
        acr,
        stream,
        tokens=tokens,
        stopwords=stop,
        mode="rtl_scan",
        allow_upper_on_stop=False,
        allow_lower_on_non_stop=is_mixed_case_acronym(acr),
        lowercase_prefix_exception=False,  # don’t need it here usually
    )

    if hit is None:
        # your fallback relax:
        hit = align_acronym_to_initials(
            acr,
            stream,
            tokens=tokens,
            stopwords=stop,
            mode="rtl_scan",
            allow_upper_on_stop=True,
            allow_lower_on_non_stop=is_mixed_case_acronym(acr),
            lowercase_prefix_exception=False,
        )
    if hit is None:
        return []

    tok_left = hit.tok_left
    tok_right = len(tokens) - 1  # anchored to end in this matcher

    if acr and acr[0].isdigit():
        tok_left = consume_left_numeric_designator(acr=acr, tokens=tokens, tok_left=tok_left)

    tok_left, tok_right = expand_numeric_leading_window(tokens, tok_left, tok_right)

    hit_tokens = hit.hit_tokens

    if acr_starts_with_digit:
        tok_left = consume_left_numeric_designator(acr=acr, tokens=tokens, tok_left=tok_left)

    # Expand window to include adjacent numeric-leading tokens (e.g., "3M")
    def _numeric_leading(idx: int) -> bool:
        init = first_alnum_char_upper(tokens[idx])
        return (init is not None) and (not init.isalpha())

    # 5) Build kept phrase inside the window: matched tokens + bridges + numeric-leading
    phrase = build_kept_phrase(
        tokens,
        tok_left=tok_left,
        tok_right=tok_right,
        hit_tokens=hit_tokens,
        bridges=bridges,
        include_numeric_leading=True,
        first_alnum_char_upper=first_alnum_char_upper,
    )
    if not phrase:
        return []

    # 6) Tight character spans over the ORIGINAL snippet
    base = m.start("pre")  # where `pre` starts in `snippet`
    ds = base + starts[tok_left]
    de = base + ends[tok_right]

    # 7) Normalize for display; indices stay tight
    norm = normalize_definition(phrase)
    if not norm:
        return []
    raw_window = collapse_ws(snippet[ds:de])  # raw chars between ds..de (just whitespace-collapsed)

    disp = normalize_definition(phrase)
    if not disp:
        return []
    return [LocalDefMatch(def_start=ds, def_end=de, definition=disp, raw=raw_window)]
