## Makefile

Common commands (see `make help`):

- `make run` — FastAPI with reload at http://localhost:8000
- `make lint` — Ruff
- `make typecheck` — mypy
- `make test` — unit tests + coverage
- `make test-integration` — spins up Postgres and runs `-m integration`
- `make migrate` / `make downgrade` — Alembic migrations
- `make build` — Docker image with OCI labels
- `make ci-local` — lint + typecheck + test + build
- `make release VERSION=vX.Y.Z` — tag & push, update `CHANGELOG.md`

## Environment Variables

| Variable                     | Local                                | Staging          | Production       | Description                    |
|-----------------------------|--------------------------------------|------------------|------------------|--------------------------------|
| APP_ENV                     | `local`                              | `staging`        | `production`     | Runtime env                    |
| PORT                        | `8000`                               | `8080`           | `8080`           | HTTP port                      |
| LOG_LEVEL                   | `debug`                              | `info`           | `info`           | Logging level                  |
| DATABASE_URL                | `postgresql+psycopg://user:pass@localhost:5432/unacronym` | **secret** | **secret** | Postgres DSN                   |
| MAX_BODY_BYTES              | `2000000`                            | `1000000`        | `1000000`        | Request body limit             |
| RESOLVE_TIMEOUT_MS          | `2000`                               | `2000`           | `2000`           | Resolver timeout               |
| DATABASE_DISABLED               | `true`                               | `false`          | `false`          | Disable auth (dev only)        |
| API_KEY_HEADER              | `X-API-Key`                          | `X-API-Key`      | `X-API-Key`      | API key header                 |
| SENTRY_DSN                  | _empty_                              | **secret**       | **secret**       | Error monitoring               |
| ENABLE_DOCS                 | `true`                               | `true`           | `false`          | Swagger/Redoc availability     |
| API_KEY_HASH_SCHEME         | `argon2id`                           | `argon2id`       | `argon2id`       | Secret hashing                 |
| API_KEY_CACHE_TTL_SECONDS   | `0`                                  | `60`             | `60`             | Key cache TTL (s)              |
| REQUEST_TIMEOUT_MS          | `2000`                               | `2000`           | `2000`           | Request timeout                |

Copy `.env.example` → `.env` (or `.env.local`) and tweak for your environment.

## Services
Offsets use Python-slice semantics: start inclusive, end exclusive.

Path versioning: breaking changes ⇒ /v2/...; additive only in /v1/....

meta.model_version mirrors plainera-core resolver version.

Headers: X-Request-Id, X-Input-Bytes, X-Body-Limit-Bytes. X-RateLimit-* reserved.
