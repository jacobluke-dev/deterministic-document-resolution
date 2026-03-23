from __future__ import annotations

from dataclasses import dataclass

import public_api.core.services.resolve_service as resolve_service_mod
import pytest
from fastapi import status
from public_api.core.services.resolve_service import ResolveError, _lang_from_locale
from public_api.schemas.error import ErrorCode
from public_api.schemas.resolve import ResolveOptions, ResolveRequest


@dataclass(frozen=True)
class _Occurrence:
    acronym: str
    start_offset: int
    end_offset: int

class _LockedSemaphore:
    def locked(self) -> bool:
        return True


class _UnlockedSemaphore:
    def locked(self) -> bool:
        return False


@pytest.mark.parametrize(
    ("locale", "expected"),
    [
        ("en-GB", "en"),
        ("en-US", "en"),
        ("", "en"),
    ],
)
def test_lang_from_locale(locale: str, expected: str):
    assert _lang_from_locale(locale) == expected


class TestValidateAndPrepare:
    def test_validate_and_prepare_rejects_whitespace_only_text(self, service_factory):
        svc, _ = service_factory()
        payload = ResolveRequest(text="   \n\t   ", options=None)

        with pytest.raises(ResolveError) as exc:
            svc._validate_and_prepare(payload)

        assert exc.value.http_status == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert exc.value.code == ErrorCode.UNPROCESSABLE_ENTITY
        assert exc.value.message == "Text must not be empty."
        assert exc.value.details == {"hint": "Provide non-empty 'text'"}

    def test_validate_and_prepare_rejects_oversized_text(self, monkeypatch, service_factory):
        svc, _ = service_factory()
        monkeypatch.setattr(resolve_service_mod, "TEXT_MAX_LEN", 5)

        payload = ResolveRequest(text="abcdef", options=None)

        with pytest.raises(ResolveError) as exc:
            svc._validate_and_prepare(payload)

        assert exc.value.http_status == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        assert exc.value.code == ErrorCode.PAYLOAD_TOO_LARGE
        assert exc.value.message == "Body/text too large."
        assert exc.value.details == {"limit": 5, "actual": 6}

    def test_validate_and_prepare_supplies_default_options(self, service_factory):
        svc, _ = service_factory()
        payload = ResolveRequest(text="ABC means Alpha Beta Company.", options=None)

        opts, lang = svc._validate_and_prepare(payload)

        assert opts.model_dump() == ResolveOptions.model_validate({}).model_dump()
        assert lang == _lang_from_locale(opts.locale)

    def test_validate_and_prepare_extracts_lang_from_locale(self, service_factory, opts_factory):
        svc, _ = service_factory()
        payload = ResolveRequest(
            text="ABC means Alpha Beta Company.",
            options=opts_factory(locale="en-GB"),
        )

        opts, lang = svc._validate_and_prepare(payload)

        assert opts.locale == "en-GB"
        assert lang == "en"



class TestRaiseIfOverloaded:
    def test_raise_if_overloaded_raises_503_when_locked(self, service_factory):
        svc, _ = service_factory(semaphore=_LockedSemaphore())

        with pytest.raises(ResolveError) as exc:
            svc._raise_if_overloaded()

        assert exc.value.http_status == status.HTTP_503_SERVICE_UNAVAILABLE
        assert exc.value.code == ErrorCode.SERVICE_UNAVAILABLE
        assert exc.value.details == {"reason": "OVERLOADED"}


def test_raise_if_overloaded_noops_when_unlocked(service_factory):
    svc, _ = service_factory(semaphore=_UnlockedSemaphore())

    svc._raise_if_overloaded()  # no exception
