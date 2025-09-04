from unittest import mock

# Adjust this import to your actual module path
# e.g. from plainera_core.db_manager.sessions import make_async_sessionmaker
import plainera_core.db_manager.sessions as sessions
import pytest


@pytest.mark.unit
class TestMakeAsyncSessionmaker:
    def test_builds_engine_and_sessionmaker_with_expected_args(self, monkeypatch):
        url = "postgresql+asyncpg://u:p@localhost:5432/db"

        fake_engine = object()
        fake_sessionmaker = object()

        m_create_async_engine = mock.Mock(return_value=fake_engine)
        m_async_sessionmaker = mock.Mock(return_value=fake_sessionmaker)

        # Patch the symbols inside the module under test
        monkeypatch.setattr(sessions, "create_async_engine", m_create_async_engine)
        monkeypatch.setattr(sessions, "async_sessionmaker", m_async_sessionmaker)

        rv = sessions.make_async_sessionmaker(url)

        # Engine created with desired options
        m_create_async_engine.assert_called_once()
        args, kwargs = m_create_async_engine.call_args
        assert args == (url,)
        assert kwargs["pool_pre_ping"] is True
        assert kwargs["connect_args"] == {"server_settings": {"search_path": "unacronym"}}

        # async_sessionmaker called with correct engine and config
        m_async_sessionmaker.assert_called_once_with(fake_engine, expire_on_commit=False)

        # Factory returned to caller
        assert rv is fake_sessionmaker
