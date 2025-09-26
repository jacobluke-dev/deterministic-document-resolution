from dataclasses import dataclass
from typing import Callable

from plainera_unacronym.nlp.common.constants import DEFAULT_STOPWORDS, BRIDGES_DEFAULT


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    inline_cues: tuple[str, ...]
    parenthetical_allows: tuple[Callable[[str, str], bool], ...] = ()
    # Phrase limits / toggles
    max_phrase_chars: int = 200
    enabled_parenthetical: bool = True
    enabled_inline: bool = True
    stop: frozenset[str] = frozenset(DEFAULT_STOPWORDS)
    bridges: frozenset[str] = frozenset(BRIDGES_DEFAULT)

    # Acronym policy
    min_acr_len: int = 2
    max_acr_len: int = 10
    acr_allowed: str = r"A-Z0-9&./-"

    # Confidence weights
    conf_parenthetical: float = 0.95
    conf_inline: float = 0.80

    # Inline cue regex fragments (case-insensitive)
    inline_cues: tuple[str, ...] = (
        r"short\s+for",
        r"stands?\s+for",
        r"is\s+(?:an\s+)?acronym\s+for",
    )

    # Optional stricter gating
    require_two_words: bool = True

    # Optional plugin names (must be registered in plugins.registry)
    plugins: tuple[str, ...] = ()
