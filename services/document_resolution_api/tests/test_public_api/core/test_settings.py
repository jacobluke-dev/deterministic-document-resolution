import importlib
import sys
import types

import pytest


class TestAppSettings:
    def _reload(self, monkeypatch, env: dict[str, str] | None = None):
        # stop .env from repopulating vars
        monkeypatch.setitem(
            sys.modules,
            "dotenv",
            types.SimpleNamespace(load_dotenv=lambda *a, **k: False),
        )
        # clear app & db env, then set desired
        for k in ("APP_ENV","PORT","LOG_LEVEL","CORS_ORIGINS","MAX_BODY_BYTES",
                  "ENABLE_DOCS","REQUEST_TIMEOUT_MS","DEFAULT_LOCALE",
                  "DEFAULT_WINDOW_CHARS","MAX_INFLIGHT","SENTRY_DSN",
                  "RUN_DB_MIGRATIONS","DATABASE_URL","DB_SCHEMA","DATABASE_DISABLED"):
            monkeypatch.delenv(k, raising=False)
        if env:
            for k, v in env.items():
                monkeypatch.setenv(k, str(v))
        import public_api.core.settings as settings
        return importlib.reload(settings)

    def test_defaults(self, monkeypatch):
        settings = self._reload(monkeypatch)
        s = settings.app_settings
        assert s.APP_ENV == "development"
        assert s.PORT == 8000
        assert s.LOG_LEVEL == "INFO"
        assert s.cors_origins == []
        assert s.MAX_BODY_BYTES == 2 * 1024 * 1024
        assert s.ENABLE_DOCS is True
        assert s.REQUEST_TIMEOUT_MS == 5000
        assert s.DEFAULT_LOCALE == "en-GB"
        assert s.DEFAULT_WINDOW_CHARS == 120
        assert s.MAX_INFLIGHT == 0
        assert s.SENTRY_DSN is None
        assert s.RUN_DB_MIGRATIONS is True

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("", []),
            ("https://a.com", ["https://a.com"]),
            (" http://x, https://y , ws://z ", ["http://x", "https://y", "ws://z"]),
        ],
    )
    def test_cors_origins_parsing(self, monkeypatch, raw, expected):
        settings = self._reload(monkeypatch, {"CORS_ORIGINS": raw})
        assert settings.app_settings.cors_origins == expected


class TestDatabaseSettings:
    def _reload(self, monkeypatch, env: dict[str, str] | None = None):
        # stop .env from repopulating vars
        import sys
        import types
        monkeypatch.setitem(
            sys.modules,
            "dotenv",
            types.SimpleNamespace(load_dotenv=lambda *a, **k: False),
        )
        for k in ("DATABASE_URL","DB_SCHEMA","APP_ENV","DATABASE_DISABLED"):
            monkeypatch.delenv(k, raising=False)
        if env:
            for k, v in env.items():
                monkeypatch.setenv(k, str(v))
        import public_api.core.settings as settings
        return importlib.reload(settings)

    def test_db_defaults(self, monkeypatch):
        settings = self._reload(monkeypatch)
        ds = settings.db_settings
        assert ds.APP_ENV == "local"
        assert ds.DATABASE_DISABLED is False
        assert ds.DB_SCHEMA == "document_resolution"
        # Naming convention has expected keys
        for k in ("ix","uq","ck","fk","pk"):
            assert k in ds.NAMING_CONVENTION

    def test_database_url_missing_raises(self, monkeypatch):
        settings = self._reload(monkeypatch)  # no DATABASE_URL in env
        with pytest.raises(RuntimeError) as ei:
            _ = settings.db_settings.database_url
        # Helpful message includes path hint
        msg = str(ei.value)
        assert "DATABASE_URL missing" in msg
        assert ".env" in msg

    def test_database_url_present_returns_str(self, monkeypatch):
        # Use a valid AnyUrl for Pydantic; psycopg driver string
        url = "postgresql+psycopg://user:pass@localhost:5432/document_resolution"
        settings = self._reload(monkeypatch, {"DATABASE_URL": url})
        assert settings.db_settings.database_url == url
