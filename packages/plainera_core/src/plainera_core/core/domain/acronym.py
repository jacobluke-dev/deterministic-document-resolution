from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Acronym:
    text: str
