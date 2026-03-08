import re

from plainera_unacronym.nlp.common.types import pattern_cache, AcronymDetectorConfig


def compile_acronym_pattern(cfg: AcronymDetectorConfig) -> re.Pattern[str]:
    """
    Compile a linear, low-backtracking token pattern for acronym-like candidates.

    The pattern is assembled from multiple branches (separators, dotted forms, compact caps,
    digit-prefixed, and optional mixed-case) and wrapped in word boundaries to avoid
    matching inside longer identifiers.

    Args:
        cfg (AcronymDetectorConfig): Detector configuration controlling bounds and enabled branches.

    Returns:
        re.Pattern[str]: Compiled regex with a named group "tok" for candidate spans.
    """
    # Cache key must include all switches that change the pattern’s shape.
    # NOTE: if our config field is named `enable_mixed_case` (no underscore),
    key = (cfg.min_len, cfg.max_len, cfg.allow_chars, cfg.enable_dotted, cfg.enable_mixed_case)
    if key in pattern_cache:
        return pattern_cache[key]

    # Escape the set of allowed internal separators for the character class.
    sep = re.escape(cfg.allow_chars)

    # 1) Chunks with internal separators (R&D, USB-C, O’RAN, I/O).
    #    - Letters/digits on both sides of a separator from cfg.allow_chars.
    #    - Optional whitespace around the separator is allowed (e.g., "R & D").
    with_seps = rf"(?:[A-Z0-9]+(?:\s*[{sep}]\s*[A-Z0-9]+)+)"

    # 2) Dotted initialisms (opt-in).
    #    - One or more "LETTER + dot" pairs *followed by a final LETTER*.
    #      Ending on a letter keeps the right-hand \b boundary valid.
    #      Examples matched: "U.S", "U.S.A"  (the trailing period, if any, is
    #      left outside the match and later trimmed by strip_trailing_punct()).
    dotted = r"(?:[A-Z]\.)+[A-Z]"

    # 3) Compact ALL-CAPS/alnum runs within length bounds.
    #    - First char must be A–Z; remaining are A–Z or 0–9.
    #    - Length bounds derive from cfg.{min,max}_len (inclusive) and apply to
    #      the whole run (the first char counts toward the total).
    tail_min = max(cfg.min_len - 1, 0)
    tail_max = max(cfg.max_len - 1, 0)
    tail_max = max(tail_max, tail_min)
    compact = rf"(?:[A-Z][A-Z0-9]{{{tail_min},{tail_max}}})"

    # 4) digit-prefixed compact, e.g. 3GPP, 2FA, 5G, 80211AX (if you allow those)
    # Ensure there's at least one letter after the digit run: [0-9]+[A-Z]
    dmin = max(cfg.min_len - 2, 0)
    dmax = max(cfg.max_len - 2, 0)
    dmax = max(dmax, dmin)
    digit_compact = rf"(?:[0-9]+[A-Z][A-Z0-9]{{{dmin},{dmax}}})"

    # 5) CamelCaps (opt-in, upper-first) for brand-style abbreviations.
    #    - Simple, linear pattern that captures tokens like "TfL", "eBPF" (upper-first only here).
    #    - We also guard this in the iterator by relaxing the caps ratio only if ≥2 uppers exist.
    camel_uc = r"(?:[A-Z][a-z]?){2,5}"

    # 6) lower-prefix mixed-case, e.g. mRNA, eBPF, iOS, miRNA
    # - 1-2 lowercase letters prefix
    # - then at least 2 uppercase letters somewhere to avoid matching normal words
    # - allow trailing digits (optional)
    lower_prefix_mixed = r"(?:[a-z]{1,2}[A-Z]{2,}[A-Za-z0-9]*)"

    # 6b) lower-prefix brand-style camel, e.g. eBay, iPhone, eBook
    # - 1-2 lowercase letters
    # - 1 uppercase letter
    # - then 1+ lowercase letters (prevents matching "eBPF" which is handled by lower_prefix_mixed)
    # - optional trailing alnum
    lower_prefix_brand = r"(?:[a-z]{1,2}[A-Z][a-z]+[A-Za-z0-9]*)"

    # 6c)
    # Upper-prefix mixed-case, e.g. LaTeX, PowerBI, OpenAI (if you want), iPhoneOS-style variants
    upper_prefix_mixed = r"(?:[A-Z][a-z]{1,}[A-Z][A-Za-z0-9]*)"

    # 7)
    # ALL-CAPS (or alnum) with an optional short lowercase suffix (e.g. PDFs, GPUs, NHSs).
    # Keep suffix short to avoid normal words; 1–3 is usually enough.
    caps_with_suffix = r"(?:[A-Z]{2,}[a-z]{1,3})"

    # Order matters: keep more specific branches (with_seps/dotted) before the generic compact.
    branches = [with_seps, caps_with_suffix, compact, digit_compact]

    if cfg.enable_dotted:
        branches.insert(1, dotted)  # give dotted precedence over compact
    if cfg.enable_mixed_case:
        branches.append(camel_uc)
        branches.append(lower_prefix_mixed)
        branches.append(upper_prefix_mixed)
        branches.append(lower_prefix_brand)

    # Word boundaries prevent matching inside longer identifiers/words.
    # The branches themselves include internal punctuation; \b only applies at edges.
    token = r"\b(?P<tok>" + "|".join(branches) + r")\b"

    pat = re.compile(token)
    pattern_cache[key] = pat
    return pat
