import logging
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


@pytest.fixture
def _patch(monkeypatch):
    def _apply(func, **replacements):
        g = func.__globals__
        for name, impl in replacements.items():
            monkeypatch.setitem(g, name, impl)
        return func  # optional convenience
    return _apply

class DummyCfgCls:
    def __init__(self, max_phrase_chars=80):
        self.max_phrase_chars = max_phrase_chars
        self.require_initials_match = False


@pytest.fixture
def dummy_cfg():
    return DummyCfgCls
