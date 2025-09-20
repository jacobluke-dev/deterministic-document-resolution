
from typing import Mapping, Optional, Literal

from . import ExtractionConfig
from .extract_first_occ import extract_near_firsts
from .util import picks_from_global
from .. import DetectorConfig
from ..common.types import FirstOccurrence


Strategy = Literal["anchored", "global", "hybrid"]

def extract_definitions(
    text: str,
    *,
    firsts: Optional[Mapping[str, FirstOccurrence]] = None,
    cfg: ExtractionConfig = ExtractionConfig(),
    strategy: Strategy = "anchored",
) -> dict[str, Optional[dict]]:
    """
    Returns a map: normalized_key -> {definition, acr_span, def_span, confidence, original_definition} | None
    """
    det_cfg = DetectorConfig()
    if strategy == "global" or not firsts:
        # Build once, then choose nearest per key (uses internal helper)
        return picks_from_global(text, firsts, det_cfg=det_cfg, ext_cfg=cfg)
    if strategy == "anchored":
        return extract_near_firsts(text, firsts, cfg=cfg)
    # hybrid: anchored first, and only for None keys run one global pass to try to fill gaps
    anchored = extract_near_firsts(text, firsts, cfg=cfg)
    if not anchored or all(v is not None for v in anchored.values()):
        return anchored
    global_picks = picks_from_global(text, firsts, det_cfg=det_cfg, ext_cfg=cfg)
    return {k: anchored.get(k) or global_picks.get(k) for k in set(anchored) | set(global_picks)}
