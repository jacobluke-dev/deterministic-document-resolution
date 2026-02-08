import pytest
from plainera_unacronym.nlp.common.shared import normalize_acronym_key
from plainera_unacronym.nlp.common.types import DetectorConfig, FirstOccurrence, Occurrence


@pytest.fixture
def cfg() -> DetectorConfig:
    return DetectorConfig()


@pytest.fixture
def fo():
    def _fo(cfg: DetectorConfig, acr: str, s: int, e: int, conf: float = 0.9) -> FirstOccurrence:
        k = normalize_acronym_key(acr, cfg.allow_chars, dotted_mode=cfg.dotted_display)
        assert k
        return FirstOccurrence(acronym=acr, start_offset=s, end_offset=e, confidence=conf, normalized_key=k)
    return _fo


@pytest.fixture
def occ():
    def _occ(cfg: DetectorConfig, acr: str, s: int, e: int, conf: float = 0.9) -> Occurrence:
        k = normalize_acronym_key(acr, cfg.allow_chars, dotted_mode=cfg.dotted_display)
        return Occurrence(
            acronym=acr,
            start_offset=s,
            end_offset=e,
            confidence=conf,
            context_window=(max(0, s - 20), e + 20),
            normalized_key=k,
            reasons=None,
        )
    return _occ
