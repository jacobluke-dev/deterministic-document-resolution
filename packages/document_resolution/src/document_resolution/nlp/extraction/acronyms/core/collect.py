def initials_match(acr: str, phrase: str) -> bool:
    """Check if an acronym fits the phrase's initials as an ordered subsequence.

    Args:
      acr (str): The Acronym to test.
      phrase (str): Candidate long-form phrase used to derive initials.

    Returns:
      bool: True if the acronym's letters appear in order within the phrase initials;
      otherwise False.

    """
    parts: list[str] = []
    for w in phrase.split():
        if not w:
            continue
        if not w[0].isalpha():
            continue

        # Expand ALL-CAPS alphabetic tokens (RNA -> RNA), otherwise first-letter only.
        if w.isalpha() and w.isupper() and len(w) > 1:
            parts.append(w)
        else:
            parts.append(w[0].upper())

    initials = "".join(parts).upper()

    j = 0
    for ch in acr:
        if ch.isalpha():
            pos = initials.find(ch.upper(), j)
            if pos == -1:
                return False
            j = pos + 1
    return True
