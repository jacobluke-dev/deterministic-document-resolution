from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["AcronymDetector", "DetectorConfig", "FirstOccurrence", "Occurrence"]


def __getattr__(name: str):
    if name == "AcronymDetector":
        from plainera_unacronym.nlp.detection.acronym.detector import AcronymDetector

        return AcronymDetector
    if name in {"DetectorConfig", "FirstOccurrence", "Occurrence"}:
        from plainera_unacronym.nlp.common.types import DetectorConfig, FirstOccurrence, Occurrence

        return {"DetectorConfig": DetectorConfig, "FirstOccurrence": FirstOccurrence, "Occurrence": Occurrence}[name]
    raise AttributeError(name)


if TYPE_CHECKING:
    from plainera_unacronym.nlp.common.types import DetectorConfig, FirstOccurrence, Occurrence
    from plainera_unacronym.nlp.detection.acronym.detector import AcronymDetector
