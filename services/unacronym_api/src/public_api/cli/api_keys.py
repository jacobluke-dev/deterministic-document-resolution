import argparse
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import text

from plainera_core.db_manager.factory import make_dbm
from public_api.core.auth.api_keys import (
    generate_key,
    hash_secret,
    parse_api_key,
)
from public_api.core.settings import app_settings


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _allow_prefixes() -> set[str]:
    raw = app_settings.API_KEY_PREFIX_ALLOWLIST or "live,test"
    return {p.strip() for p in raw.split(",") if p.strip()}


@dataclass(frozen=True)
class _KeyRef:
    """Normalised reference to an API key record."""
    key_id: str
    prefix: str | None = None


def _parse_key_ref(value: str) -> _KeyRef:
    """
    Accept either:
      - key_id (e.g. 'xgX3UcuD0rkA')
      - full key string (e.g. 'uak_test_xgX3UcuD0rkA_....')
    """
    allow = _allow_prefixes()
    parsed = parse_api_key(value, allow_prefixes=allow)
    if parsed is not None:
        return _KeyRef(key_id=parsed.key_id, prefix=parsed.prefix)
    return _KeyRef(key_id=value, prefix=None)


def cmd_create(args: argparse.Namespace) -> None:
    allow = _allow_prefixes()
    if args.prefix not in allow:
        raise SystemExit(f"Invalid --prefix '{args.prefix}'. Allowed: {', '.join(sorted(allow))}")

    dbm = make_dbm(test_mode=False)

    key_id, secret, full = generate_key(args.prefix)
    scheme = app_settings.API_KEY_HASH_SCHEME or "argon2id"
    key_hash = hash_secret(secret, scheme=scheme)

    stmt = text(
        """
        INSERT INTO unacronym.api_keys
          (key_id, key_hash, name, prefix, scopes, is_active, created_at)
        VALUES
          (:key_id, :key_hash, :name, :prefix, :scopes, true, now())
        """
    )

    with dbm.session() as s:
        s.execute(
            stmt,
            {
                "key_id": key_id,
                "key_hash": key_hash,
                "name": args.name,
                "prefix": args.prefix,
                "scopes": args.scopes or [],
            },
        )
        s.commit()

    print(full)  # shown once


def cmd_list(_: argparse.Namespace) -> None:
    dbm = make_dbm(test_mode=False)
    stmt = text(
        """
        SELECT key_id, prefix, name, is_active, created_at, last_used_at, expires_at
        FROM unacronym.api_keys
        ORDER BY created_at DESC NULLS LAST
        """
    )
    with dbm.session() as s:
        rows = s.execute(stmt).mappings().all()

    for r in rows:
        print(
            f"{r['prefix']}\t{r['key_id']}\tactive={r['is_active']}\t"
            f"name={r['name']}\tlast_used={r['last_used_at']}\texpires={r['expires_at']}"
        )


def cmd_revoke(args: argparse.Namespace) -> None:
    ref = _parse_key_ref(args.key)

    dbm = make_dbm(test_mode=False)
    stmt = text(
        """
        UPDATE unacronym.api_keys
        SET is_active = false,
            expires_at = COALESCE(expires_at, now())
        WHERE key_id = :key_id
        """
    )
    with dbm.session() as s:
        res = s.execute(stmt, {"key_id": ref.key_id})
        s.commit()

    # res.rowcount is driver-dependent but usually works for UPDATE
    print(f"revoked {ref.key_id}")


def cmd_rotate(args: argparse.Namespace) -> None:
    old = _parse_key_ref(args.key)

    # Create new key first
    args_create = argparse.Namespace(prefix=args.prefix, name=args.name, scopes=args.scopes)
    cmd_create(args_create)

    if args.revoke_old:
        cmd_revoke(argparse.Namespace(key=old.key_id))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="api-keys")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("create")
    c.add_argument("--prefix", default="test")
    c.add_argument("--name", default=None)
    c.add_argument("--scopes", nargs="*", default=[])
    c.set_defaults(func=cmd_create)

    l = sub.add_parser("list")
    l.set_defaults(func=cmd_list)

    r = sub.add_parser("revoke")
    r.add_argument("key", help="key_id or full key string (uak_...)")
    r.set_defaults(func=cmd_revoke)

    rot = sub.add_parser("rotate")
    rot.add_argument("key", help="old key_id or full key string (uak_...)")
    rot.add_argument("--prefix", default="test")
    rot.add_argument("--name", default=None)
    rot.add_argument("--scopes", nargs="*", default=[])
    rot.add_argument("--revoke-old", action="store_true", help="revoke the old key after creating a new one")
    rot.set_defaults(func=cmd_rotate)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
