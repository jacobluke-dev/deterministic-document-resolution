# tests/test_plainera_unacronym/test_nlp/plugins/test_activation.py

import plainera_unacronym.nlp.plugins.activation as activation_mod
from plainera_unacronym.nlp.plugins.activation import _safe_sniff, autodetect_domains
from plainera_unacronym.nlp.types import DetectorConfig

# --------- Test doubles (plugins) ---------------------------------------------

class GoodBioPlugin:
    name = "bio"
    def sniff(self, text: str) -> bool:
        return "mRNA" in text

class FalsePlugin:
    name = "other"
    def sniff(self, text: str) -> bool:
        return False

class ErrorPlugin:
    name = "error"
    def sniff(self, text: str) -> bool:
        raise ValueError("boom")

class NoSniffPlugin:
    """No sniff() method; autodetect should ignore it."""
    name = "nosniff"


class TestSafeSniff:
    def test_returns_true_when_plugin_sniff_true(self):
        assert _safe_sniff(GoodBioPlugin(), "mRNA present") is True

    def test_returns_false_when_plugin_sniff_false(self):
        assert _safe_sniff(FalsePlugin(), "no signals") is False

    def test_swallow_exception_and_return_false(self):
        assert _safe_sniff(ErrorPlugin(), "anything") is False


class TestAutodetectDomains:
    def test_detects_plugins_that_sniff_true(self, monkeypatch):
        cfg = DetectorConfig()
        monkeypatch.setattr(
            activation_mod, "DOMAIN_PLUGINS",
            {"bio": GoodBioPlugin(), "other": FalsePlugin()},
            raising=False,
        )
        detected = autodetect_domains("text with mRNA marker", cfg)
        assert detected == frozenset({"bio"})

    def test_ignores_plugins_without_sniff(self, monkeypatch):
        cfg = DetectorConfig()
        monkeypatch.setattr(
            activation_mod, "DOMAIN_PLUGINS",
            {"nosniff": NoSniffPlugin()},
            raising=False,
        )
        detected = autodetect_domains("mRNA present", cfg)
        assert detected == frozenset()

    def test_handles_exceptions_and_detects_others(self, monkeypatch):
        cfg = DetectorConfig()
        monkeypatch.setattr(
            activation_mod, "DOMAIN_PLUGINS",
            {"bio": GoodBioPlugin(), "error": ErrorPlugin()},
            raising=False,
        )
        detected = autodetect_domains("mRNA here", cfg)
        assert detected == frozenset({"bio"})

    def test_respects_cap_truncation(self, monkeypatch):
        cfg = DetectorConfig()
        monkeypatch.setattr(
            activation_mod, "DOMAIN_PLUGINS",
            {"bio": GoodBioPlugin()},
            raising=False,
        )
        long_text = ("x" * 100_000) + "mRNA"  # signal after default cap
        assert autodetect_domains(long_text, cfg) == frozenset()

        # Increase cap so the signal is visible
        assert autodetect_domains(long_text, cfg, cap=200_000) == frozenset({"bio"})

    def test_returns_frozenset(self, monkeypatch):
        cfg = DetectorConfig()
        monkeypatch.setattr(
            activation_mod, "DOMAIN_PLUGINS",
            {"bio": GoodBioPlugin()},
            raising=False,
        )
        out = autodetect_domains("mRNA", cfg)
        assert isinstance(out, frozenset)
