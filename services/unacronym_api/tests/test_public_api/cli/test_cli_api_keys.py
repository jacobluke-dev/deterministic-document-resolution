from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import public_api.cli.api_keys as cli
import pytest


@dataclass
class _FakeSession:
    executed: list[tuple[Any, dict[str, Any]]]
    committed: bool = False
    rows: list[dict[str, Any]] | None = None

    def execute(self, stmt: Any, params: dict[str, Any] | None = None) -> Any:
        self.executed.append((stmt, params or {}))

        class _Res:
            def mappings(self_inner) -> _Res:
                return self_inner

            def all(self_inner) -> list[dict[str, Any]]:
                return list(self.rows or [])

        return _Res()

    def commit(self):
        self.committed = True


class _FakeDBM:
    def __init__(self):
        self.session_obj = _FakeSession(executed=[])

    @contextmanager
    def session(self):
        yield self.session_obj


class _Settings:
    API_KEY_PREFIX_ALLOWLIST = "live,test"
    API_KEY_HASH_SCHEME = "argon2id"


def test_allow_prefixes_from_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cli, "app_settings", _Settings)
    assert cli._allow_prefixes() == {"live", "test"}


def test_parse_key_ref_raw_id(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cli, "app_settings", _Settings)

    # Make parse_api_key return None so it falls back to raw id
    monkeypatch.setattr(cli, "parse_api_key", lambda *_args, **_kw: None)

    ref = cli._parse_key_ref("xgX3UcuD0rkA")
    assert ref.key_id == "xgX3UcuD0rkA"
    assert ref.prefix is None


def test_parse_key_ref_full_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cli, "app_settings", _Settings)

    class _Parsed:
        key_id = "abc123DEF"
        prefix = "test"

    monkeypatch.setattr(cli, "parse_api_key", lambda *_args, **_kw: _Parsed())

    ref = cli._parse_key_ref("uak_test_abc123DEF_secretstuff")
    assert ref.key_id == "abc123DEF"
    assert ref.prefix == "test"


def test_parse_hash_scheme_valid():
    assert cli.parse_hash_scheme(None) == "argon2id"
    assert cli.parse_hash_scheme("argon2id") == "argon2id"
    assert cli.parse_hash_scheme("BCRYPT") == "bcrypt"


def test_parse_hash_scheme_invalid():
    with pytest.raises(ValueError):
        cli.parse_hash_scheme("md5")


def test_cmd_create_rejects_invalid_prefix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cli, "app_settings", _Settings)

    args = argparse.Namespace(prefix="nope", name=None, scopes=[])
    with pytest.raises(SystemExit):
        cli.cmd_create(args)


def test_cmd_create_inserts_and_prints_full_key(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(cli, "app_settings", _Settings)

    fake_dbm = _FakeDBM()
    monkeypatch.setattr(cli, "make_dbm", lambda test_mode=False: fake_dbm)

    monkeypatch.setattr(cli, "generate_key", lambda prefix: ("kid123", "sec456", "uak_test_kid123_sec456"))
    monkeypatch.setattr(cli, "hash_secret", lambda secret, scheme: f"HASH({scheme})::{secret}")

    args = argparse.Namespace(prefix="test", name="my key", scopes=["read", "write"])
    cli.cmd_create(args)

    # assert DB execute called with expected params
    sess = fake_dbm.session_obj
    assert sess.committed is True
    assert len(sess.executed) == 1
    _stmt, params = sess.executed[0]
    assert params["key_id"] == "kid123"
    assert params["key_hash"] == "HASH(argon2id)::sec456"
    assert params["name"] == "my key"
    assert params["prefix"] == "test"
    assert params["scopes"] == ["read", "write"]

    out = capsys.readouterr().out.strip()
    assert out == "uak_test_kid123_sec456"


def test_cmd_revoke_updates_and_prints(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(cli, "app_settings", _Settings)

    fake_dbm = _FakeDBM()
    monkeypatch.setattr(cli, "make_dbm", lambda test_mode=False: fake_dbm)

    # Normalise key ref to key_id
    monkeypatch.setattr(cli, "_parse_key_ref", lambda value: cli._KeyRef(key_id="kid123", prefix=None))

    args = argparse.Namespace(key="kid123")
    cli.cmd_revoke(args)

    sess = fake_dbm.session_obj
    assert sess.committed is True
    assert len(sess.executed) == 1
    _stmt, params = sess.executed[0]
    assert params["key_id"] == "kid123"

    out = capsys.readouterr().out.strip()
    assert out == "revoked kid123"


def test_cmd_rotate_creates_then_revokes_when_flag_set(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, Any]] = []

    monkeypatch.setattr(cli, "_parse_key_ref", lambda value: cli._KeyRef(key_id="oldkid", prefix="test"))

    def _fake_create(ns: argparse.Namespace):
        calls.append(("create", (ns.prefix, ns.name, tuple(ns.scopes))))

    def _fake_revoke(ns: argparse.Namespace):
        calls.append(("revoke", ns.key))

    monkeypatch.setattr(cli, "cmd_create", _fake_create)
    monkeypatch.setattr(cli, "cmd_revoke", _fake_revoke)

    args = argparse.Namespace(
        key="oldkid",
        prefix="test",
        name="rotated",
        scopes=["s1"],
        revoke_old=True,
    )
    cli.cmd_rotate(args)

    assert calls == [
        ("create", ("test", "rotated", ("s1",))),
        ("revoke", "oldkid"),
    ]


def test_cmd_rotate_does_not_revoke_when_flag_not_set(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []

    monkeypatch.setattr(cli, "_parse_key_ref", lambda value: cli._KeyRef(key_id="oldkid", prefix="test"))
    monkeypatch.setattr(cli, "cmd_create", lambda ns: calls.append("create"))
    monkeypatch.setattr(cli, "cmd_revoke", lambda ns: calls.append("revoke"))

    args = argparse.Namespace(
        key="oldkid",
        prefix="test",
        name=None,
        scopes=[],
        revoke_old=False,
    )
    cli.cmd_rotate(args)

    assert calls == ["create"]
