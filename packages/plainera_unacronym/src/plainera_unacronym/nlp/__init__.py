from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "AcronymDetector",
    "DefinedTermDetector",
    "AcronymDetectorConfig",
    "DefinedTermDetectorConfig",
    "FirstOccurrence",
    "Occurrence",
]

def __getattr__(name: str):
    if name == "AcronymDetector":
        from plainera_unacronym.nlp.detection.acronym.detector import AcronymDetector

        return AcronymDetector
    if name == "DefinedTermDetector":
        from plainera_unacronym.nlp.detection.defined_terms import DefinedTermDetector
        return DefinedTermDetector
    if name in {"AcronymDetectorConfig", "DefinedTermDetectorConfig", "FirstOccurrence", "Occurrence"}:
        from plainera_unacronym.nlp.common.types import (AcronymDetectorConfig,
                                                         DefinedTermDetectorConfig,
                                                         FirstOccurrence,
                                                         Occurrence)

        return {
            "AcronymDetectorConfig": AcronymDetectorConfig,
            "DefinedTermDetectorConfig": DefinedTermDetectorConfig,
            "FirstOccurrence": FirstOccurrence,
            "Occurrence": Occurrence,
        }[name]
    raise AttributeError(name)


if TYPE_CHECKING:
    from plainera_unacronym.nlp.common.types import (AcronymDetectorConfig,
                                                     DefinedTermDetectorConfig,
                                                     FirstOccurrence,
                                                     Occurrence)
    from plainera_unacronym.nlp.detection.acronym.detector import AcronymDetector
    from plainera_unacronym.nlp.detection.defined_terms import DefinedTermDetector
