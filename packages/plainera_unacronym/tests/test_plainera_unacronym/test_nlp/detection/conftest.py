from plainera_unacronym.nlp import AcronymDetectorConfig
import pytest

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
