from document_resolution.nlp.extraction.acronyms.matchers.defs.constants import AlignmentHit, InitialsStream


def _align_rtl_scan_wrapper(
    acronym_alignment: list[str],
    *,
    stream: InitialsStream,
    allow_upper_on_stop: bool,
    allow_lower_on_non_stop: bool,
) -> AlignmentHit | None:
    """Align acronym targets to the stream using right-to-left scanning.

    This is a thin wrapper around `align_rtl_scan(...)` that converts the matched
    stream-letter positions into token-span metadata (`AlignmentHit`).

    Args:
        acronym_alignment: Acronym alignment targets (typically from `acr_alignment_targets`), in the
            order expected by the RTL scan aligner.
        stream: Initials stream providing `letters`, `owners`, and per-letter stopword
            status `is_stop`.
        allow_upper_on_stop: If True, permit uppercase targets to match initials owned
            by stopword tokens.
        allow_lower_on_non_stop: If True, permit lowercase targets to match initials
            owned by non-stopword tokens.

    Returns:
        An `AlignmentHit` if a match is found, else None. The hit's token bounds are
        derived from the owning tokens of the matched stream positions.
    """
    used = align_rtl_scan(
        acronym_alignment,
        stream.letters,
        stream.is_stop,
        allow_upper_on_stop=allow_upper_on_stop,
        allow_lower_on_non_stop=allow_lower_on_non_stop,
    )
    if used is None:
        return None

    used = list(used)
    hit_tokens = {stream.owners[p] for p in used}
    return AlignmentHit(
        used_letter_pos=used,
        hit_tokens=hit_tokens,
        tok_left=min(hit_tokens),
        tok_right=max(hit_tokens),
    )


def align_rtl_scan(
    targets: list[str],
    initials: list[str],
    is_stop_letter: list[bool],
    *,
    allow_upper_on_stop: bool,
    allow_lower_on_non_stop: bool = False,
) -> list[int] | None:
    """Align acronym targets right-to-left against an initials stream scanned in order.

    This performs a greedy scan over `initials` from left to right, while consuming
    `targets` from right to left. A target character matches when the uppercased
    target equals the current initials letter and stopword constraints are satisfied.

    Stopword constraints:
      - Lowercase target (e.g. 'o' in "MofM") prefers a stopword owner letter. It may
        align to non-stop letters only if `allow_lower_on_non_stop=True`.
      - Uppercase target prefers a non-stop owner letter. It may align to stop letters
        only if `allow_upper_on_stop=True`.

    Args:
        targets: Acronym alignment targets (caseful). Lowercase indicates a stopword
            preference; uppercase indicates a non-stopword preference.
        initials: Initials letters in scan order (expected uppercase).
        is_stop_letter: Parallel to `initials`; True if the owning token is a stopword.
        allow_upper_on_stop: If True, allow uppercase targets to align onto stop letters.
        allow_lower_on_non_stop: If True, allow lowercase targets to align onto non-stop letters.

    Returns:
        A list of indices into `initials` representing the matched letters (in scan order),
        or None if the full target sequence cannot be matched.
    """
    ti = len(targets) - 1  # target index (RTL)
    li = 0  # initials scan index
    used: list[int] = []

    while li < len(initials) and ti >= 0:
        need = targets[ti].upper()

        if initials[li] == need:
            want_stop = targets[ti].islower()

            if want_stop:
                ok = is_stop_letter[li] or allow_lower_on_non_stop
            else:
                ok = (not is_stop_letter[li]) or allow_upper_on_stop

            if ok:
                used.append(li)
                ti -= 1

        li += 1

    return None if ti >= 0 else used


def _acronym_letters_rtl(tok: str) -> list[str]:
    """Extract acronym characters in right-to-left order.

    Strips light trailing/leading punctuation, keeps only alphanumeric characters,
    uppercases letters, and returns them in RTL order.

    Args:
        tok: Token potentially containing an acronym/initialism (e.g. "U.S.A", "HTTP2").

    Returns:
        List of uppercased alphanumeric characters in RTL order.
        Examples:
          - "RNA" -> ["A", "N", "R"]
          - "U.S.A" -> ["A", "S", "U"]
          - "HTTP2" -> ["2", "P", "T", "T", "H"]
    """
    t = tok.strip(".,;:)]}»”'\"")
    chars = [c.upper() for c in t if c.isalnum()]
    chars.reverse()
    return chars
