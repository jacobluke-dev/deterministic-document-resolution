from typing import Callable

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
