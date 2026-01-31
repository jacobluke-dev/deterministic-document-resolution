import logging

import pytest


@pytest.fixture(autouse=True)
def _reset_plainera_logger():
    lg = logging.getLogger("plainera")

    # snapshot
    old_handlers = list(lg.handlers)
    old_level = lg.level
    old_propagate = lg.propagate
    old_disabled = lg.disabled

    # reset to a state caplog can see
    lg.handlers.clear()
    lg.propagate = True
    lg.disabled = False
    # (leave level alone unless you need to force it)

    try:
        yield
    finally:
        # restore to avoid leaking to other tests
        lg.handlers[:] = old_handlers
        lg.level = old_level
        lg.propagate = old_propagate
        lg.disabled = old_disabled


@pytest.fixture(autouse=True)
def _ensure_logging_enabled():
    prev = logging.root.manager.disable
    logging.disable(logging.NOTSET)  # re-enable everything for this test
    yield
    logging.disable(prev)


class CaptureSinkCls:
    def __init__(self):
        self.items = []

    def enqueue(self, payload):
        self.items.append(payload)

    async def enqueue_async(self, payload):
        self.items.append(payload)


@pytest.fixture
def capture_sink():
    return CaptureSinkCls
