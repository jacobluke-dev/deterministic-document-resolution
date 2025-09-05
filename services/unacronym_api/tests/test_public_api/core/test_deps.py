import importlib
from asyncio import Semaphore


class DummyResolver:
    pass


def _reload_deps(max_inflight: int | None, create_stub):
    """
    Helper: monkeypatch settings + provider and reload deps module
    so container is re-initialised fresh for each test.
    """
    if max_inflight is None:
        max_inflight = 0
    from public_api.core import providers as providers_mod
    from public_api.core import settings as settings_mod

    settings_mod.app_settings.MAX_INFLIGHT = max_inflight
    providers_mod.create_resolver = create_stub  # stub out real provider

    from public_api.core import deps as deps_mod
    return importlib.reload(deps_mod)


class TestAppContainer:

    def test_semaphore_disabled_if_zero(self):
        def stub():
            return DummyResolver()

        deps = _reload_deps(0, stub)
        assert deps.container.semaphore is None

    def test_semaphore_enabled_if_positive(self):
        def stub():
            return DummyResolver()

        deps = _reload_deps(3, stub)
        sem = deps.container.semaphore
        assert isinstance(sem, Semaphore)
        assert sem._value == 3  # CPython detail but fine for us


class TestGetResolver:
    def test_returns_same_resolver_instance(self):
        def stub():
            return DummyResolver()

        deps = _reload_deps(0, stub)
        assert deps.get_resolver() is deps.container.resolver


class TestGetSemaphore:
    def test_returns_none_when_disabled(self):
        def stub():
            return DummyResolver()

        deps = _reload_deps(0, stub)
        assert deps.get_semaphore() is None

    def test_returns_same_instance_when_enabled(self):
        def stub():
            return DummyResolver()

        deps = _reload_deps(2, stub)
        assert deps.get_semaphore() is deps.container.semaphore


class TestAppContainerIntegration:
    def test_real_container_initializes_with_provider(self, monkeypatch):
        # Use the real provider, but control MAX_INFLIGHT to avoid flakiness
        from public_api.core import deps as deps_mod
        from public_api.core import providers as providers_mod
        from public_api.core import settings as settings_mod

        settings_mod.app_settings.MAX_INFLIGHT = 1
        # ensure provider is not stubbed
        importlib.reload(providers_mod)
        # rebuild container with real provider + current settings
        deps = importlib.reload(deps_mod)

        # sanity checks: resolver exists, semaphore set to 1
        assert deps.container.resolver is not None
        sem = deps.container.semaphore
        assert isinstance(sem, Semaphore)
        assert getattr(sem, "_value", None) == 1
        # dependency functions hand back the same instances
        assert deps.get_resolver() is deps.container.resolver
        assert deps.get_semaphore() is sem
