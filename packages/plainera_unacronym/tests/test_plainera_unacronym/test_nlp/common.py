
def picked_def(extr, key: str):
    """Return extracted definition for acronym key if present, else None."""
    pick = extr.picks.get(key)
    if pick is None:
        return None
    return pick.definition
