import pytest


class FakeRegistry:
    def __init__(self, runner):
        self._runner = runner

    def get(self, key):
        return self._runner

@pytest.fixture
def fake_registry():
    def make(runner):
        return FakeRegistry(runner)
    return make
