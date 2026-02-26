from __future__ import annotations

import argparse
from datetime import datetime, timezone

from sqlalchemy import text

from plainera_core.db_manager.factory import make_dbm
from public_api.core.auth.api_keys import generate_key, hash_secret


def _now() -> datetime:
    return datetime.now(timezone.utc)


def cmd_create(args: argparse.Namespace) -> None:
    dbm = make_dbm(test_mode=False)
    key_id, secret, full = generate_key(args.prefix)
    key_hash = hash_secret(secret, scheme="argon2id")

    stmt = text(
        """
        INSERT INTO unacronym.api_keys (key_id, key_hash, name, prefix, scopes, is_active, created_at)
        VALUES (:key_id, :key_hash, :name, :prefix, :scopes, true, now())
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
    dbm = make_dbm(test_mode=False)
    stmt = text(
        """
        UPDATE unacronym.api_keys
        SET is_active = false
        WHERE key_id = :key_id
        """
    )
    with dbm.session() as s:
        s.execute(stmt, {"key_id": args.key_id})
        s.commit()
    print(f"revoked {args.key_id}")


def cmd_rotate(args: argparse.Namespace) -> None:
    # Create a new key then optionally revoke old
    args_create = argparse.Namespace(prefix=args.prefix, name=args.name, scopes=args.scopes)
    cmd_create(args_create)
    if args.revoke_old:
        cmd_revoke(argparse.Namespace(key_id=args.revoke_old))


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
    r.add_argument("key_id")
    r.set_defaults(func=cmd_revoke)

    rot = sub.add_parser("rotate")
    rot.add_argument("--prefix", default="test")
    rot.add_argument("--name", default=None)
    rot.add_argument("--scopes", nargs="*", default=[])
    rot.add_argument("--revoke-old", default=None)
    rot.set_defaults(func=cmd_rotate)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
