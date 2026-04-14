from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import public_api.core.auth.api_keys as auth_mod
import pytest
from document_resolution_core.db_manager.connection import DBManager
from public_api.core.auth.api_keys import parse_api_key
from public_api.db.models import ApiKey


def test_parse_accepts_valid():
    raw = "uak_live_AbCDef1234_abcdefghijklmnopqrstuvwxyzABCDEFGHijklmnop123456"
    out = parse_api_key(raw, allow_prefixes={"live", "test"})
    assert out is not None
    assert out.prefix == "live"
    assert out.key_id == "AbCDef1234"


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "uak__x_y",
        "uak_badprefix_x_y",
        "uak_live_short_short",
        "uak_live_@@@_secret",
    ],
)
def test_parse_rejects_malformed(raw):
    out = parse_api_key(raw, allow_prefixes={"live", "test"})
    assert out is None


def _seed_api_key(
    session_factory,
    *,
    key_id: str = "test_key_id",
    prefix: str = "test",
    is_active: bool = True,
    expires_at: datetime | None = None,
    daily_quota: int | None = None,
    scopes: list[str] | None = None,
) -> ApiKey:
    with session_factory() as s:
        row = ApiKey(
            user_id=None,
            key_id=key_id,
            key_hash="dummy_hash",
            name="test key",
            prefix=prefix,
            scopes=scopes or [],
            is_active=is_active,
            expires_at=expires_at,
            daily_quota=daily_quota,
        )
        s.add(row)
        s.commit()
        s.refresh(row)
        return row


class TestFetchKeyRecord:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        auth_mod._CACHE.clear()

    def test_returns_record_on_cache_miss(self, session_factory):
        seeded = _seed_api_key(
            session_factory,
            key_id="abc123",
            prefix="test",
            daily_quota=25,
            scopes=["resolve:read"],
        )

        dbm = auth_mod.make_dbm(test_mode=True) if hasattr(auth_mod, "make_dbm") else None
        if dbm is None:
            bind = session_factory.kw["bind"]
            dbm = DBManager(engine=bind, session_factory=session_factory)

        rec = auth_mod.fetch_key_record(dbm, "abc123")

        assert rec is not None
        assert rec.id == seeded.id
        assert rec.user_id is None
        assert rec.prefix == "test"
        assert rec.key_hash == "dummy_hash"
        assert rec.scopes == ("resolve:read",)
        assert rec.is_active is True
        assert rec.daily_quota == 25

    def test_returns_none_when_missing(self, session_factory):

        bind = session_factory.kw["bind"]
        dbm = DBManager(engine=bind, session_factory=session_factory)

        rec = auth_mod.fetch_key_record(dbm, "does_not_exist")

        assert rec is None

    def test_returns_none_when_inactive(self, session_factory):

        _seed_api_key(
            session_factory,
            key_id="inactive_key",
            is_active=False,
        )

        bind = session_factory.kw["bind"]
        dbm = DBManager(engine=bind, session_factory=session_factory)

        rec = auth_mod.fetch_key_record(dbm, "inactive_key")

        assert rec is None

    def test_returns_none_when_expired(self, session_factory):

        _seed_api_key(
            session_factory,
            key_id="expired_key",
            expires_at=datetime.now(UTC) - timedelta(days=1),
        )

        bind = session_factory.kw["bind"]
        dbm = DBManager(engine=bind, session_factory=session_factory)

        rec = auth_mod.fetch_key_record(dbm, "expired_key")

        assert rec is None

    def test_caches_record_after_first_lookup(self, session_factory, monkeypatch):

        monkeypatch.setattr(auth_mod.app_settings, "API_KEY_CACHE_TTL_SECONDS", 60)
        auth_mod._CACHE.clear()

        seeded = _seed_api_key(
            session_factory,
            key_id="cached_key",
            daily_quota=10,
        )

        bind = session_factory.kw["bind"]
        dbm = DBManager(engine=bind, session_factory=session_factory)

        rec1 = auth_mod.fetch_key_record(dbm, "cached_key")
        assert rec1 is not None
        assert rec1.id == seeded.id

        def boom():
            raise AssertionError("DB session should not be used on cache hit")

        monkeypatch.setattr(dbm, "session", boom)

        rec2 = auth_mod.fetch_key_record(dbm, "cached_key")

        assert rec2 is not None
        assert rec2.id == seeded.id


    def test_daily_quota_none_round_trips(self, session_factory):

        _seed_api_key(
            session_factory,
            key_id="no_quota_key",
            daily_quota=None,
        )

        bind = session_factory.kw["bind"]
        dbm = DBManager(engine=bind, session_factory=session_factory)

        rec = auth_mod.fetch_key_record(dbm, "no_quota_key")

        assert rec is not None
        assert rec.daily_quota is None

    def test_returns_cached_record_without_db_hit(self, session_factory, _patch):

        bind = session_factory.kw["bind"]
        dbm = DBManager(engine=bind, session_factory=session_factory)

        cached = auth_mod.ApiKeyRecord(
            id=123,
            user_id=None,
            prefix="test",
            key_hash="cached_hash",
            scopes=("resolve:read",),
            is_active=True,
            expires_at=None,
            daily_quota=42,
        )

        class DummyCache:
            def get(self, key):
                assert key == "cached_key"
                return cached

            def put(self, key, value):
                raise AssertionError("put should not be called on cache hit")

        _patch(auth_mod.fetch_key_record, _CACHE=DummyCache())

        rec = auth_mod.fetch_key_record(dbm, "cached_key")

        assert rec is cached

    def test_cache_miss_puts_record(self, session_factory, _patch):
        key_id = f"abc123_{uuid.uuid4().hex[:8]}"

        _seed_api_key(
            session_factory,
            key_id=key_id,
            daily_quota=25,
            scopes=["resolve:read"],
        )

        bind = session_factory.kw["bind"]
        dbm = DBManager(engine=bind, session_factory=session_factory)

        seen = {}

        class DummyCache:
            def get(self, key):
                return None

            def put(self, key, value):
                seen["key"] = key
                seen["value"] = value

        _patch(auth_mod.fetch_key_record, _CACHE=DummyCache())

        rec = auth_mod.fetch_key_record(dbm, key_id)

        assert rec is not None
        assert rec.daily_quota == 25
        assert seen["key"] == key_id
        assert seen["value"] == rec
