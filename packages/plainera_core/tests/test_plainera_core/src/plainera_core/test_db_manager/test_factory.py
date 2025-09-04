import types
from unittest import mock

# Adjust import path if your factory lives elsewhere
import plainera_core.db_manager.factory as factory
import pytest
from public_api.core.settings import db_settings
from sqlalchemy import text


@pytest.mark.unit
class TestMakeDbm:
    def _setup_mocks(self, monkeypatch):
        """
        Patch create_engine, sessionmaker, DBManager, and db_settings.
        """
        # Sentinels
        fake_engine = object()
        fake_session_factory = object()

        # create_engine mock
        m_create_engine = mock.Mock(return_value=fake_engine)
        monkeypatch.setattr(factory, "create_engine", m_create_engine)

        # sessionmaker mock
        m_sessionmaker = mock.Mock(return_value=fake_session_factory)
        monkeypatch.setattr(factory, "sessionmaker", m_sessionmaker)

        # DBManager mock that captures args
        class FakeDBM:
            def __init__(self, *, engine, session_factory):
                self.engine = engine
                self.session_factory = session_factory

        monkeypatch.setattr(factory, "DBManager", FakeDBM)

        # db_settings mock with default URL
        fake_settings = types.SimpleNamespace(database_url="postgresql://default/db")
        monkeypatch.setattr(factory, "db_settings", fake_settings)

        return m_create_engine, m_sessionmaker, FakeDBM, fake_engine, fake_session_factory

    def test_uses_override_url_when_provided(self, monkeypatch, dbm):
        m_create_engine, m_sessionmaker, FakeDBM, fake_engine, fake_session_factory = self._setup_mocks(monkeypatch)

        override = "postgresql://override/db"
        dbm = factory.make_dbm(url=override, test_mode=False)

        # create_engine called with override DSN
        m_create_engine.assert_called_once()
        args, kwargs = m_create_engine.call_args
        assert args[0] == override
        assert kwargs["pool_pre_ping"] is True
        assert kwargs["future"] is True
        # Current implementation passes None explicitly
        assert "poolclass" in kwargs and kwargs["poolclass"] is None

        # sessionmaker bound to the engine
        m_sessionmaker.assert_called_once_with(bind=fake_engine, autoflush=False, autocommit=False, future=True)

        # DBManager constructed with engine + session factory
        assert isinstance(dbm, FakeDBM)
        assert dbm.engine is fake_engine
        assert dbm.session_factory is fake_session_factory

    def test_uses_db_settings_when_url_not_provided(self, monkeypatch):
        m_create_engine, m_sessionmaker, FakeDBM, fake_engine, fake_session_factory = self._setup_mocks(monkeypatch)

        dbm = factory.make_dbm()  # no url passed

        m_create_engine.assert_called_once()
        args, _ = m_create_engine.call_args
        assert args[0] == factory.db_settings.database_url  # pulled from settings
        assert isinstance(dbm, FakeDBM)

    @pytest.mark.parametrize("test_mode, expected_poolclass", [(True, factory.NullPool), (False, None)])
    def test_poolclass_varies_with_test_mode(self, monkeypatch, test_mode, expected_poolclass):
        m_create_engine, *_ = self._setup_mocks(monkeypatch)

        factory.make_dbm(test_mode=test_mode)

        _, kwargs = m_create_engine.call_args
        # Ensure poolclass aligns with flag
        assert "poolclass" in kwargs
        assert kwargs["poolclass"] is expected_poolclass


@pytest.fixture
def ensure_schema():
    schema = db_settings.DB_SCHEMA

    def _apply(conn_or_session):
        # Connection has exec_driver_sql; Session uses execute(text(...))
        try:
            conn_or_session.exec_driver_sql(f'SET search_path TO "{schema}", public')
        except AttributeError:
            conn_or_session.execute(text(f'SET search_path TO "{schema}", public'))
    return _apply


@pytest.mark.integration
class TestEngineAndSessionFixtures:
    def test_engine_has_expected_search_path(self, engine_factory, apply_migrations_once, ensure_schema):
        with engine_factory.connect() as conn:
            ensure_schema(conn)
            sp = conn.exec_driver_sql("SHOW search_path").scalar()
            assert db_settings.DB_SCHEMA in sp, f"search_path={sp!r}"

    def test_session_basic_query(self, session_factory, apply_migrations_once, ensure_schema):
        Session = session_factory
        with Session() as s:
            ensure_schema(s)
            one = s.execute(text("SELECT 1")).scalar_one()
            assert one == 1


@pytest.mark.integration
class TestDbmFixture:
    def test_dbm_can_open_session_and_query(self, dbm, apply_migrations_once, ensure_schema):
        with dbm.session() as s:
            ensure_schema(s)
            val = s.execute(text("SELECT 1")).scalar_one()
            assert val == 1

    def test_dbm_search_path_visible(self, dbm, apply_migrations_once, ensure_schema):
        with dbm.session() as s:
            ensure_schema(s)
            sp = s.execute(text("SHOW search_path")).scalar_one()
            assert db_settings.DB_SCHEMA in sp, f"search_path={sp!r}"
