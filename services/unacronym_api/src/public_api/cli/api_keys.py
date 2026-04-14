"""
Administrative CLI for managing public API keys.

This module provides operator-facing commands to create, list, revoke, and rotate
API keys stored in ``unacronym.api_keys``. Secrets are generated client-side by
the CLI, hashed before persistence, and only shown once at creation time.

Supported operations:
  - ``create``: generate a new API key, hash the secret, persist the record, and
    print the full key exactly once.
  - ``list``: display non-secret metadata for existing keys.
  - ``revoke``: deactivate an existing key and stamp ``expires_at``.
  - ``rotate``: create a replacement key and optionally revoke the old one.

Key handling model:
  - The database stores ``key_id`` + hashed secret (``key_hash``), never the raw
    full key.
  - Full keys are parsed and validated against the configured prefix allowlist.
  - Hashing uses the scheme configured by ``API_KEY_HASH_SCHEME``.

Operational notes:
  - Output from ``create`` is sensitive and should be treated like a credential.
  - This CLI is intended for trusted administrative use, not end-user workflows.
  - Database access is performed via ``make_dbm(test_mode=False)``.

Typical usage:
  - ``api-keys create --prefix test --name "Local dev key"``
  - ``api-keys list``
  - ``api-keys revoke <key_id-or-full-key>``
  - ``api-keys rotate <key_id-or-full-key> --revoke-old``
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from document_resolution_core.db_manager.factory import make_dbm
from public_api.core.auth.api_keys import (
    generate_key,
    hash_secret,
    parse_api_key,
)
from public_api.core.settings import app_settings
from sqlalchemy import text


def _now() -> datetime:
    """
    Return the current UTC time as a timezone-aware `datetime`.

    Centralises "now" so timestamps are consistent and easy to patch in tests.

    Returns:
      A timezone-aware UTC `datetime`.
    """
    return datetime.now(timezone.utc)


def _allow_prefixes() -> set[str]:
    """
    Compute the allowed API key prefixes from application settings.

    Uses `API_KEY_PREFIX_ALLOWLIST` when configured; otherwise defaults to
    `{"live", "test"}`. Whitespace is trimmed and empty entries are ignored.

    Returns:
      A set of allowed key prefixes.
    """
    raw = app_settings.API_KEY_PREFIX_ALLOWLIST or "live,test"
    return {p.strip() for p in raw.split(",") if p.strip()}


@dataclass(frozen=True)
class _KeyRef:
    """Normalised reference to an API key record."""
    key_id: str
    prefix: str | None = None


def _parse_key_ref(value: str) -> _KeyRef:
    """
    Normalise a CLI key reference into a `key_id` (+ optional prefix).

    Accepts either a raw `key_id` (e.g. "xgX3UcuD0rkA") or a full API key string
    (e.g. "uak_test_xgX3UcuD0rkA_..."). If a full key is provided, it is parsed
    and validated against the allowed prefix set.

    Args:
      value: A raw key id or a full API key string.

    Returns:
      A `_KeyRef` containing the key id and (when parseable) the prefix.

    Notes:
      - If parsing fails, the input is treated as a raw key id.
      - Prefix allowlisting is enforced only when the value looks like a full key.
    """
    allow = _allow_prefixes()
    parsed = parse_api_key(value, allow_prefixes=allow)
    if parsed is not None:
        return _KeyRef(key_id=parsed.key_id, prefix=parsed.prefix)
    return _KeyRef(key_id=value, prefix=None)


HashScheme = Literal["argon2id", "bcrypt"]


def parse_hash_scheme(value: str | None) -> HashScheme:
    """
    Normalise API key hash scheme configuration to a supported literal.

    Accepts a potentially-untrusted string (e.g. from env/config) and returns a
    supported scheme identifier, defaulting to "argon2id".
    """
    v = (value or "argon2id").strip().lower()
    if v in ("argon2id", "bcrypt"):
        return v  # type: ignore[return-value]
    raise ValueError(f"Unsupported API_KEY_HASH_SCHEME: {value!r}")


def cmd_create(args: argparse.Namespace) -> None:
    """
    Create a new API key and persist its hash to the database.

    Validates the requested prefix, generates a new key (id + secret), hashes the
    secret using the configured scheme, inserts a new `api_keys` record, and
    prints the full key string exactly once for the operator to copy.

    Args:
      args: CLI arguments namespace containing `prefix`, `name`, and `scopes`.

    Raises:
      SystemExit: If `args.prefix` is not in the allowed prefix set.

    Notes:
      - The printed key is the only time the secret is available; it cannot be
        recovered from the stored hash.
      - `scopes` defaults to an empty list and is stored as provided.
    """
    allow = _allow_prefixes()
    if args.prefix not in allow:
        raise SystemExit(f"Invalid --prefix '{args.prefix}'. Allowed: {', '.join(sorted(allow))}")

    dbm = make_dbm(test_mode=False)

    key_id, secret, full = generate_key(args.prefix)
    scheme = parse_hash_scheme(app_settings.API_KEY_HASH_SCHEME or "argon2id")
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
    """
    List API keys from the database in a human-readable format.

    Loads key metadata (id, prefix, active flag, created/last_used/expires times)
    and prints one tab-separated line per key, ordered by most-recent creation.

    Args:
      _: Unused CLI args namespace (accepted for argparse compatibility).

    Notes:
      - Output is intended for operator inspection and scripting; the format is
        stable-ish but not a public contract.
      - This command does not reveal any secret material.
    """
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
    """
    Revoke an API key by marking it inactive and setting an expiry timestamp.

    Accepts either a raw `key_id` or a full key string; normalises to a key id,
    then updates the corresponding record to `is_active=false` and sets
    `expires_at` to `now()` if it was previously unset.

    Args:
      args: CLI arguments namespace containing `key` (key_id or full key string).

    Notes:
      - This is an idempotent-ish operation: re-revoking a key will keep it
        inactive and preserve the earliest `expires_at` once set.
      - The update is performed by `key_id` only; prefix is not required.
    """
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
        s.execute(stmt, {"key_id": ref.key_id})
        s.commit()

    print(f"revoked {ref.key_id}")


def cmd_rotate(args: argparse.Namespace) -> None:
    """
    Rotate an API key by creating a replacement and optionally revoking the old.

    Creates a new key using the provided prefix/name/scopes, prints the new full
    key string, and (optionally) revokes the old key if `--revoke-old` is set.

    Args:
      args: CLI arguments namespace containing the old `key` plus new key options.

    Notes:
      - The new key is created before revoking the old key to reduce downtime.
      - Revocation targets the old key by `key_id` even if the user provided a
        full key string.
    """
    old = _parse_key_ref(args.key)

    # Create new key first
    args_create = argparse.Namespace(prefix=args.prefix, name=args.name, scopes=args.scopes)
    cmd_create(args_create)

    if args.revoke_old:
        cmd_revoke(argparse.Namespace(key=old.key_id))


def main(argv: list[str] | None = None) -> None:
    """
    CLI entrypoint for managing API keys (create/list/revoke/rotate).

    Defines argparse subcommands and dispatches to the relevant command handler.
    Intended for operator use in secure environments.

    Args:
      argv: Optional argument vector (defaults to `sys.argv[1:]` via argparse).

    Notes:
      - `create` prints a full key once; treat stdout as sensitive.
      - Database connectivity is provided via `make_dbm(test_mode=False)`.
    """
    p = argparse.ArgumentParser(prog="api-keys")
    sub = p.add_subparsers(dest="cmd", required=True)

    c_parser = sub.add_parser("create")
    c_parser.add_argument("--prefix", default="test")
    c_parser.add_argument("--name", default=None)
    c_parser.add_argument("--scopes", nargs="*", default=[])
    c_parser.set_defaults(func=cmd_create)

    l_parser = sub.add_parser("list")
    l_parser.set_defaults(func=cmd_list)

    r_parser = sub.add_parser("revoke")
    r_parser.add_argument("key", help="key_id or full key string (uak_...)")
    r_parser.set_defaults(func=cmd_revoke)

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
