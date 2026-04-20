from document_resolution.nlp.extraction.acronyms.matchers.defs.constants import InitialsStream, AlignmentHit
from document_resolution.nlp.common.types import Span

def _alignment_letter_ok(
    alignment_letters: list[str],
    ai: int,
    tok_idx: int,
    *,
    tokens: list[str],
    is_stop_token: list[bool],
    allow_upper_on_stop: bool,
    allow_lower_on_non_stop: bool,
    lowercase_prefix_exception: bool,
) -> bool:
    """Check whether a single aligned letter may map to a given token.

    Implements the stopword/case constraints described in `_align_ltr_min_window`.

    Args:
        alignment_letters: Original acronym letters (may include lowercase).
        ai: Index into `alignment_letters` for the letter being matched.
        tok_idx: Candidate token index being hit.
        tokens: Original token list.
        is_stop_token: Stopword flags per token.
        allow_upper_on_stop: Allow uppercase letters to hit stop tokens.
        allow_lower_on_non_stop: Allow lowercase letters to hit non-stop tokens.
        lowercase_prefix_exception: Allow narrow exception for mixed-case leading lowercase.

    Returns:
        bool: True if mapping is allowed, False otherwise.
    """
    want_stop = alignment_letters[ai].islower()

    if not want_stop:
        return (not is_stop_token[tok_idx]) or allow_upper_on_stop

    # want stopword
    if is_stop_token[tok_idx]:
        return True
    if not allow_lower_on_non_stop:
        return False

    # narrow exception: mixed-case leading lowercase (mRNA / iOS)
    if not lowercase_prefix_exception or ai != 0 or tok_idx != 0:
        return False

    tok0 = tokens[0]
    if tok0.isalpha() and tok0.isupper() and len(tok0) > 1:
        return False

    return tok0[:1].lower() == alignment_letters[0].lower()

def _align_ltr_min_window(
    alignment_letters: list[str],
    *,
    stream: InitialsStream,
    tokens: list[str],
    is_stop_token: list[bool],
    allow_upper_on_stop: bool,
    allow_lower_on_non_stop: bool,
    lowercase_prefix_exception: bool,
) -> AlignmentHit | None:
    """Align acronym letters to a initials stream using a minimal token-span strategy.

    Scans the stream left-to-right and tries to match `alignment_letters` in order.
    Among all valid matches, selects the one that minimises `(tok_right - tok_left)`
    where token bounds are derived from `stream.owners`.

    Case/stopword constraints:
      - Uppercase target letters prefer non-stopword tokens unless `allow_upper_on_stop`.
      - Lowercase target letters prefer stopword tokens; mapping to non-stopwords is
        only allowed when `allow_lower_on_non_stop`, and can be further restricted
        by `lowercase_prefix_exception` for mixed-case acronyms (e.g. iOS/mRNA).

    Args:
        alignment_letters: Acronym letters to align (may include lowercase to signal
            stopword preference). Must be alphanumeric-only upstream.
        stream: Prebuilt initials stream (letters are expected uppercase).
        tokens: Original token list the stream was derived from.
        is_stop_token: Parallel list to `tokens` indicating stopword status per token.
        allow_upper_on_stop: If True, allow uppercase target letters to align onto
            stopword tokens.
        allow_lower_on_non_stop: If True, allow lowercase target letters to align onto
            non-stopword tokens (subject to `lowercase_prefix_exception`).
        lowercase_prefix_exception: If True, allow a narrow exception for a leading
            lowercase letter in a mixed-case acronym to map to token0.

    Returns:
        An `AlignmentHit` describing the chosen alignment, or `None` if no valid
        alignment exists.
    """
    L = [c.upper() for c in alignment_letters]  # stream letters are uppercase

    best_used: list[int] | None = None
    best_span: Span | None = None  # (tok_left, tok_right)

    for li in range(len(stream.letters)):
        if (len(stream.letters) - li) < len(L):
            break

        used = _try_align_from_ltr_start(
            alignment_letters,
            L,
            li,
            stream=stream,
            tokens=tokens,
            is_stop_token=is_stop_token,
            allow_upper_on_stop=allow_upper_on_stop,
            allow_lower_on_non_stop=allow_lower_on_non_stop,
            lowercase_prefix_exception=lowercase_prefix_exception,
        )
        if not used:
            continue

        tok_left, tok_right = _token_span_for_used(used, stream.owners)

        if best_span is None or (tok_right - tok_left) < (best_span[1] - best_span[0]):
            best_span = (tok_left, tok_right)
            best_used = used

    if not best_used or best_span is None:
        return None

    hit_tokens = {stream.owners[p] for p in best_used}
    tok_left, tok_right = best_span
    return AlignmentHit(
        used_letter_pos=best_used,
        hit_tokens=hit_tokens,
        tok_left=tok_left,
        tok_right=tok_right,
    )

def _token_span_for_used(used_letter_pos: list[int], owners: list[int]) -> Span:
    """Compute (tok_left, tok_right) from used stream letter positions.
    """
    tok_left = min(owners[p] for p in used_letter_pos)
    tok_right = max(owners[p] for p in used_letter_pos)
    return tok_left, tok_right


def _try_align_from_ltr_start(
    alignment_letters: list[str],
    L: list[str],
    li: int,
    *,
    stream: InitialsStream,
    tokens: list[str],
    is_stop_token: list[bool],
    allow_upper_on_stop: bool,
    allow_lower_on_non_stop: bool,
    lowercase_prefix_exception: bool,
) -> list[int] | None:
    """Attempt to align from a specific LTR starting stream index.

    Scans `stream.letters[li:]` and tries to match the acronym letters `L` in order.
    Applies stopword/case constraints per letter. Returns the used stream positions
    for the first valid completion (minimal end position for this `li`).

    Args:
        alignment_letters: Original acronym letters (may include lowercase).
        L: Uppercased version of `alignment_letters`.
        li: Starting index into `stream.letters`.
        stream: Prebuilt initials stream.
        tokens: Original token list.
        is_stop_token: Stopword flags per token.
        allow_upper_on_stop: Allow uppercase letters to hit stop tokens.
        allow_lower_on_non_stop: Allow lowercase letters to hit non-stop tokens.
        lowercase_prefix_exception: Allow narrow exception for mixed-case leading lowercase.

    Returns:
        list[int] | None: Used stream letter positions if a full match is found, else None.
    """
    ai = 0
    used_letter_pos: list[int] = []

    for lj in range(li, len(stream.letters)):
        if stream.letters[lj] != L[ai]:
            continue

        tok_idx = stream.owners[lj]
        if not _alignment_letter_ok(
            alignment_letters,
            ai,
            tok_idx,
            tokens=tokens,
            is_stop_token=is_stop_token,
            allow_upper_on_stop=allow_upper_on_stop,
            allow_lower_on_non_stop=allow_lower_on_non_stop,
            lowercase_prefix_exception=lowercase_prefix_exception,
        ):
            continue

        used_letter_pos.append(lj)
        ai += 1

        if ai == len(L):
            return used_letter_pos

    return None
