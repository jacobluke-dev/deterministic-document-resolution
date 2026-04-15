from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from document_resolution_core.db_manager.connection import DBManager
from public_api.core.di.deps_auth import require_api_key
from public_api.core.services.api_abuse_protection import (
    ApiAbuseProtectionService,
    QuotaExceededError,
    RateLimitExceededError,
)
from public_api.db.models import ApiKey, ApiUsageDaily, ApiUsageMinute


def _make_dbm(session_factory) -> DBManager:
    bind = session_factory.kw["bind"]
    return DBManager(engine=bind, session_factory=session_factory)


def _seed_api_key(session_factory, *, key_id: str = "test_key_id", daily_quota: int | None = None) -> ApiKey:
    with session_factory() as s:
        row = ApiKey(
            user_id=None,
            key_id=key_id,
            key_hash="dummy",
            name="test key",
            prefix="test",
            scopes=[],
            is_active=True,
            daily_quota=daily_quota,
        )
        s.add(row)
        s.commit()
        s.refresh(row)
        return row


@pytest.fixture
def _mute_abuse_logs(monkeypatch):
    monkeypatch.setattr(ApiAbuseProtectionService, "_log_rate_limited", lambda *a, **k: None)
    monkeypatch.setattr(ApiAbuseProtectionService, "_log_quota_exceeded", lambda *a, **k: None)


class TestApiAbuseProtectionService:
    def test_increment_daily_creates_then_increments(self, session_factory, _mute_abuse_logs):
        dbm = _make_dbm(session_factory)
        key = _seed_api_key(session_factory, daily_quota=10)
        svc = ApiAbuseProtectionService(dbm=dbm, sink=None)

        usage_date = datetime(2026, 3, 14, tzinfo=UTC).date()

        with dbm.session() as s:
            first = svc._increment_daily(session=s, api_key_id=key.id, usage_date=usage_date)
            second = svc._increment_daily(session=s, api_key_id=key.id, usage_date=usage_date)

        assert first == 1
        assert second == 2

        with session_factory() as s:
            row = (
                s.query(ApiUsageDaily)
                .filter(ApiUsageDaily.api_key_id == key.id)
                .filter(ApiUsageDaily.usage_date == usage_date)
                .one()
            )
            assert row.request_count == 2

    def test_increment_minute_creates_then_increments(self, session_factory, _mute_abuse_logs):
        dbm = _make_dbm(session_factory)
        key = _seed_api_key(session_factory, key_id="minute_key", daily_quota=10)
        svc = ApiAbuseProtectionService(dbm=dbm, sink=None)

        minute_bucket = datetime(2026, 3, 14, 12, 30, tzinfo=UTC)

        with dbm.session() as s:
            first = svc._increment_minute(session=s, api_key_id=key.id, minute_bucket=minute_bucket)
            second = svc._increment_minute(session=s, api_key_id=key.id, minute_bucket=minute_bucket)

        assert first == 1
        assert second == 2

        with session_factory() as s:
            row = (
                s.query(ApiUsageMinute)
                .filter(ApiUsageMinute.api_key_id == key.id)
                .filter(ApiUsageMinute.minute_bucket == minute_bucket)
                .one()
            )
            assert row.request_count == 2

    def test_enforce_allows_within_limits(self, session_factory, monkeypatch, _patch, _mute_abuse_logs):
        dbm = _make_dbm(session_factory)
        key = _seed_api_key(session_factory, key_id="allow_key", daily_quota=10)
        svc = ApiAbuseProtectionService(dbm=dbm, sink=None)

        _patch(
            ApiAbuseProtectionService.enforce,
            app_settings=SimpleNamespace(RATE_LIMIT_PER_MIN=5),
        )
        monkeypatch.setattr(
            "public_api.core.services.api_abuse_protection.ApiAbuseProtectionService._utc_now",
            staticmethod(lambda: datetime(2026, 3, 14, 12, 0, tzinfo=UTC)),
            raising=False,
        )
        monkeypatch.setattr(ApiAbuseProtectionService, "_log_rate_limited", lambda *a, **k: None)
        monkeypatch.setattr(ApiAbuseProtectionService, "_log_quota_exceeded", lambda *a, **k: None)

        svc.enforce(api_key_id=key.id, daily_quota_override=10)

        with session_factory() as s:
            daily = s.query(ApiUsageDaily).filter(ApiUsageDaily.api_key_id == key.id).one()
            minute = s.query(ApiUsageMinute).filter(ApiUsageMinute.api_key_id == key.id).one()
            assert daily.request_count == 1
            assert minute.request_count == 1

    def test_enforce_raises_rate_limit_exceeded(self, session_factory, monkeypatch, _patch, _mute_abuse_logs):
        dbm = _make_dbm(session_factory)
        key = _seed_api_key(session_factory, key_id="rate_key", daily_quota=100)
        svc = ApiAbuseProtectionService(dbm=dbm, sink=None)

        now = datetime(2026, 3, 14, 12, 0, 15, tzinfo=UTC)

        _patch(
            ApiAbuseProtectionService.enforce,
            app_settings=SimpleNamespace(RATE_LIMIT_PER_MIN=2),
        )
        monkeypatch.setattr(
            "public_api.core.services.api_abuse_protection.ApiAbuseProtectionService._utc_now",
            staticmethod(lambda: now),
            raising=False,
        )

        svc.enforce(api_key_id=key.id, daily_quota_override=100)
        svc.enforce(api_key_id=key.id, daily_quota_override=100)

        with pytest.raises(RateLimitExceededError) as exc:
            svc.enforce(api_key_id=key.id, daily_quota_override=100)

        assert exc.value.limit == 2
        assert exc.value.current_count == 3
        assert exc.value.retry_after >= 1

        # rolled back blocked request
        with session_factory() as s:
            minute = s.query(ApiUsageMinute).filter(ApiUsageMinute.api_key_id == key.id).one()
            assert minute.request_count == 2

    def test_enforce_raises_quota_exceeded(self, session_factory, monkeypatch, _patch, _mute_abuse_logs):
        dbm = _make_dbm(session_factory)
        key = _seed_api_key(session_factory, key_id="quota_key", daily_quota=2)
        svc = ApiAbuseProtectionService(dbm=dbm, sink=None)

        _patch(
            ApiAbuseProtectionService.enforce,
            app_settings=SimpleNamespace(RATE_LIMIT_PER_MIN=100),
        )
        monkeypatch.setattr(
            "public_api.core.services.api_abuse_protection.ApiAbuseProtectionService._utc_now",
            staticmethod(lambda: datetime(2026, 3, 14, 12, 0, tzinfo=UTC)),
            raising=False,
        )

        svc.enforce(api_key_id=key.id, daily_quota_override=2)
        svc.enforce(api_key_id=key.id, daily_quota_override=2)

        with pytest.raises(QuotaExceededError) as exc:
            svc.enforce(api_key_id=key.id, daily_quota_override=2)

        assert exc.value.limit == 2
        assert exc.value.current_count == 3

        # rolled back blocked request
        with session_factory() as s:
            daily = s.query(ApiUsageDaily).filter(ApiUsageDaily.api_key_id == key.id).one()
            assert daily.request_count == 2

    def test_enforce_utc_rollover_uses_new_day(self, session_factory, monkeypatch, _patch, _mute_abuse_logs):
        dbm = _make_dbm(session_factory)
        key = _seed_api_key(session_factory, key_id="rollover_key", daily_quota=1)
        svc = ApiAbuseProtectionService(dbm=dbm, sink=None)

        _patch(
            ApiAbuseProtectionService.enforce,
            app_settings=SimpleNamespace(RATE_LIMIT_PER_MIN=100),
        )

        monkeypatch.setattr(
            "public_api.core.services.api_abuse_protection.ApiAbuseProtectionService._utc_now",
            staticmethod(lambda: datetime(2026, 3, 14, 23, 59, 30, tzinfo=UTC)),
            raising=False,
        )
        svc.enforce(api_key_id=key.id, daily_quota_override=1)

        monkeypatch.setattr(
            "public_api.core.services.api_abuse_protection.ApiAbuseProtectionService._utc_now",
            staticmethod(lambda: datetime(2026, 3, 15, 0, 0, 10, tzinfo=UTC)),
            raising=False,
        )
        svc.enforce(api_key_id=key.id, daily_quota_override=1)

        with session_factory() as s:
            rows = (
                s.query(ApiUsageDaily)
                .filter(ApiUsageDaily.api_key_id == key.id)
                .order_by(ApiUsageDaily.usage_date.asc())
                .all()
            )
            assert len(rows) == 2
            assert rows[0].request_count == 1
            assert rows[1].request_count == 1


def _seed_real_api_key(session_factory, *, daily_quota: int | None = None) -> tuple[ApiKey, str]:
    """
    Seed a key row and return the full header value to use in requests.
    """
    from public_api.core.auth.api_keys import generate_key

    key_id, secret, full = generate_key("test")

    with session_factory() as s:
        row = ApiKey(
            user_id=None,
            key_id=key_id,
            key_hash="dummy",
            name="test key",
            prefix="test",
            scopes=[],
            is_active=True,
            daily_quota=daily_quota,
        )
        s.add(row)
        s.commit()
        s.refresh(row)
        return row, full


class TestResolveAbuseProtectionIntegration:

    @pytest.mark.anyio
    async def test_daily_quota_exceeded_returns_403(self, client, session_factory, monkeypatch, _patch):
        _patch(
            require_api_key,
            parse_hash_scheme=lambda *_a, **_k: "plain",
            verify_secret=lambda presented, stored, scheme=None: True,
        )

        _, full_api_key = _seed_real_api_key(session_factory, daily_quota=2)

        payload = {"text": "Alpha Beta Charlie (ABC)."}

        r1 = await client.post("/v1/resolve", json=payload, headers={"X-API-Key": full_api_key})
        r2 = await client.post("/v1/resolve", json=payload, headers={"X-API-Key": full_api_key})
        r3 = await client.post("/v1/resolve", json=payload, headers={"X-API-Key": full_api_key})

        assert r1.status_code == 200, r1.text
        assert r2.status_code == 200, r2.text
        assert r3.status_code == 403, r3.text

        body = r3.json()
        assert body["error"] == "quota_exceeded"
        assert body["limit"] == 2
        assert body["reset_at"]

    @pytest.mark.anyio
    async def test_rate_limit_exceeded_returns_429_with_retry_after(self, client, session_factory, monkeypatch, _patch):
        _patch(
            require_api_key,
            parse_hash_scheme=lambda *_a, **_k: "plain",
            verify_secret=lambda presented, stored, scheme=None: True,
        )
        _patch(
            ApiAbuseProtectionService.enforce,
            app_settings=SimpleNamespace(RATE_LIMIT_PER_MIN=2),
        )

        _, full_api_key = _seed_real_api_key(session_factory, daily_quota=100)

        payload = {"text": "Alpha Beta Charlie (ABC)."}

        r1 = await client.post("/v1/resolve", json=payload, headers={"X-API-Key": full_api_key})
        r2 = await client.post("/v1/resolve", json=payload, headers={"X-API-Key": full_api_key})
        r3 = await client.post("/v1/resolve", json=payload, headers={"X-API-Key": full_api_key})
        r4 = await client.post("/v1/resolve", json=payload, headers={"X-API-Key": full_api_key})

        assert r1.status_code == 200, r1.text
        assert r2.status_code == 200, r2.text
        assert r3.status_code == 429, r3.text
        assert r4.status_code == 429, r4.text

        body = r3.json()
        assert body["error"] == "rate_limited"
        assert body["limit"] == 2
        assert body["reset_at"]
        assert "Retry-After" in r3.headers
        assert int(r3.headers["Retry-After"]) >= 1

    @pytest.mark.anyio
    async def test_daily_quota_resets_on_new_utc_day(self, client, session_factory, monkeypatch, _patch):

        _patch(
            require_api_key,
            parse_hash_scheme=lambda *_a, **_k: "plain",
            verify_secret=lambda presented, stored, scheme=None: True,
        )

        monkeypatch.setattr(
            "public_api.core.services.api_abuse_protection.ApiAbuseProtectionService._utc_now",
            staticmethod(lambda: datetime(2026, 3, 14, 23, 59, 30, tzinfo=UTC)),
            raising=False,
        )

        _, full_api_key = _seed_real_api_key(session_factory, daily_quota=2)

        payload = {"text": "Alpha Beta Charlie (ABC)."}

        r1 = await client.post("/v1/resolve", json=payload, headers={"X-API-Key": full_api_key})
        r2 = await client.post("/v1/resolve", json=payload, headers={"X-API-Key": full_api_key})
        r3 = await client.post("/v1/resolve", json=payload, headers={"X-API-Key": full_api_key})
        assert r1.status_code == 200, r1.text
        assert r2.status_code == 200, r2.text
        assert r3.status_code == 403, r3.text

        # NEXT DAY
        monkeypatch.setattr(
            "public_api.core.services.api_abuse_protection.ApiAbuseProtectionService._utc_now",
            staticmethod(lambda: datetime(2026, 3, 15, 0, 0, 10, tzinfo=UTC)),
            raising=False,
        )

        r3 = await client.post("/v1/resolve", json=payload, headers={"X-API-Key": full_api_key})
        assert r3.status_code == 200, r3.text
