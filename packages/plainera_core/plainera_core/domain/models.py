from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Acronym:
    text: str


@dataclass(frozen=True, slots=True)
class DefinitionCandidate:
    text: str
    score: float = 0.0
