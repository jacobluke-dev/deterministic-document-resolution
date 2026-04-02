from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

from plainera_unacronym.nlp.common.types import Occurrence


@dataclass(frozen=True, slots=True)
class DroppedOccurrence:
    acronym: str
    start: int
    end: int
    rule: str
    detail: str


RuleFn: TypeAlias = Callable[
    [str, list[Occurrence]],
    tuple[list[Occurrence], list[DroppedOccurrence]],
]
"""
Rule function contract for post-detection cleanup.

Signature:
    (text, occs) -> (kept, dropped)

Rules must:
  - Be deterministic (no randomness, no global state).
  - Not mutate Occurrence objects in place.
  - Not assume occurrences are pre-sorted; sort internally if needed.
  - Return `kept` as the authoritative list for subsequent rules.
  - Return `dropped` with stable `rule` identifiers and a concise `detail`.
  - Accept `text` even if unused (use `_text` to mark intentional non-use).

Rule ordering is defined by RULES_TIER1.
"""
