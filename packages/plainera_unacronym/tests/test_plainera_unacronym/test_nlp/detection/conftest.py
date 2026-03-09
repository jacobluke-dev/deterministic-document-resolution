from dataclasses import replace

import pytest
from plainera_unacronym.nlp import AcronymDetectorConfig
from plainera_unacronym.nlp.common.types import DefinedTermDetectorConfig
from plainera_unacronym.nlp.detection.defined_terms import DefinedTermDetector


@pytest.fixture
def test_cfg():
    def _make_cfg(**overrides):
        defaults = {
            "allow_chars": "&/-",
            "window_chars": 80,
            "dotted_display": "strip",
            "debug_reasons": False,
            "debug_anomalies": False,
        }
        defaults.update(overrides)
        return AcronymDetectorConfig(**defaults)

    return _make_cfg


@pytest.fixture
def cfg_terms_det_factory():
    def make(**overrides) -> DefinedTermDetectorConfig:
        return replace(DefinedTermDetectorConfig(), **overrides)
    return make


@pytest.fixture
def defined_term_detector_factory(cfg_terms_det_factory):
    def make(**overrides) -> DefinedTermDetector:
        cfg = cfg_terms_det_factory(**overrides)
        return DefinedTermDetector(cfg)

    return make
