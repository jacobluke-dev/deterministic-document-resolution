from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["Detector", "DetectorConfig", "FirstOccurrence", "Occurrence"]


def __getattr__(name: str):
    if name == "Detector":
        from ..nlp.detection.detector import Detector

        return Detector
    if name in {"DetectorConfig", "FirstOccurrence", "Occurrence"}:
        from ..nlp.common.types import DetectorConfig, FirstOccurrence, Occurrence

        return {"DetectorConfig": DetectorConfig, "FirstOccurrence": FirstOccurrence, "Occurrence": Occurrence}[name]
    raise AttributeError(name)


if TYPE_CHECKING:
    from ..nlp.common.types import DetectorConfig, FirstOccurrence, Occurrence
    from ..nlp.detection.detector import Detector
