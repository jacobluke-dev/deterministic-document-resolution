from typing import Callable

import plainera_unacronym.nlp.detection.detector as det
import pytest

pytest_plugins = (
    "test_kit.fixtures",
)


@pytest.fixture
def span() -> Callable[[str, str], tuple[int, int]]:
    def _span(text: str, token: str) -> tuple[int, int]:
        s = text.index(token)
        return s, s + len(token)
    return _span


class NullSink:
    def __call__(self, *a, **k): pass
    def __getattr__(self, _): return lambda *a, **k: None

@pytest.fixture(autouse=True)
def patch_sink(monkeypatch):
    dummy = NullSink()
    monkeypatch.setattr(det, "sink", dummy, raising=True)
    yield dummy
