import pytest
from dataclasses import replace

from plainera_unacronym.nlp import AcronymDetectorConfig
from plainera_unacronym.nlp.common.types import DefinedTermDetectorConfig


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
