#### Variables ####

POETRY ?= poetry
RUN := $(POETRY) run
PORT ?= 8000
IMAGE ?= unacronym-api
APP_VERSION ?= 0.1.0
GIT_SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)

# Always use the compose file in ../
COMPOSE := docker compose -f ../docker-compose.yml

.PHONY: install run fmt lint style typecheck test cover htmlcov clean hooks \
        build all build-remove-cache up down logs nuke help migrate makemigration downgrade \
        seed-local al e mbic-help ci-local release

help:
	@echo "Common targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sed -e 's/:.*##/: /' | sort

install:
	$(POETRY) install --no-interaction

# ---- Dev & Quality ---------------------------------------------

run: ## Run app locally with reload
	PYTHONPATH=src $(RUN) uvicorn public_api.main:create_app --factory --reload --host 0.0.0.0 --port $(PORT)

fmt: ## Format with Ruff formatter
	$(RUN) ruff format src tests

lint: ## Lint (and autofix simple issues)
	$(RUN) ruff check src tests --fix

style: ## CI-friendly style check (no writes)
	$(RUN) ruff format --check src tests
	$(RUN) ruff check src tests

typecheck: ## Type checking
	$(RUN) mypy --config-file mypy.ini

test: ## Tests + coverage (quiet)
	ENVIRONMENT=TEST PYTHONPATH=src $(RUN) pytest -q \
	  --junitxml=pytest-results.xml \
	  --cov=public_api \
	  --cov-report=term \
	  --cov-report=xml:coverage.xml \
	  --cov-fail-under=80

cover: ## Verbose coverage
	PYTHONPATH=src $(RUN) pytest --cov=public_api --cov-report=term --cov-report=xml --cov-fail-under=80

htmlcov: ## HTML coverage report
	PYTHONPATH=src $(RUN) pytest --cov=public_api --cov-report=html
	@python -c "import webbrowser; webbrowser.open('htmlcov/index.html')" >/dev/null 2>&1 || true

print-env:
	@echo "PYTHONPATH: $(PYTHONPATH)"
	@env

# ---- Migrations (Alembic) --------------------------------------
alembic-help:
	$(COMPOSE) exec -T -w /app api python -m alembic -h || (echo "Alembic not available in image" && exit 1)

migrate: ## 2nd:  Apply migrations (upgrade head)
	$(COMPOSE) exec -w /app api \
	  python -m alembic -c public_api/alembic.ini upgrade head

# usage: make makemigration msg="add glossary tables"
makemigration: ## 1st: Create migration file
	@test -n "$(msg)" || (echo "Usage: make makemigration msg=\"your message\""; exit 2)
	$(COMPOSE) exec -w /app api \
	  python -m alembic -c public_api/alembic.ini revision --autogenerate -m "$(msg)"

# usage: make downgrade n=1
n ?= 1
downgrade: ## Roll back N steps (default 1)
	$(COMPOSE) exec -w /app api \
	  python -m alembic -c public_api/alembic.ini downgrade -$(n)

# ---- Seed -------------------------------------------------------
seed-local: ## Seed sample data (Docker env; APP_ENV=local)
	$(COMPOSE) exec -w /app api \
	  python -m public_api.db.seed.seed --force

# ---- Build & CI -------------------------------------------------

LABELS = --label org.opencontainers.image.title=$(IMAGE) \
         --label org.opencontainers.image.version=$(APP_VERSION) \
         --label org.opencontainers.image.revision=$(GIT_SHA)

all: build-remove-cache up

build-remove-cache:
	$(COMPOSE) down
	$(COMPOSE) build --no-cache

build:
	$(COMPOSE) build

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

nuke: ## stop & remove volumes (DATA LOSS)
	@read -p "Remove volumes (data loss)? type 'yes': " a; [ "$$a" = "yes" ]
	$(COMPOSE) down -v

logs:
	$(COMPOSE) logs -f

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build
	find src -name '__pycache__' -type d -prune -exec rm -rf {} +

ci-local: ## Simulate CI locally (lint, typecheck, test, build)
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test
	$(MAKE) build

# ---- Release ----------------------------------------------------
VERSION ?=
release: ## Tag & push (VERSION=vX.Y.Z). Generates/updates CHANGELOG.md (simple).
	@if [ -z "$(VERSION)" ]; then echo "Usage: make release VERSION=vX.Y.Z"; exit 2; fi
	@echo "## $(VERSION) - $$(date -u +%Y-%m-%d)" >> CHANGELOG.md
	@git log --pretty=format:'- %s' $$(git describe --tags --abbrev=0 2>/dev/null)..HEAD >> CHANGELOG.md || true
	git add CHANGELOG.md
	git commit -m "chore(release): $(VERSION)" || true
	git tag -a $(VERSION) -m "Release $(VERSION)"
	git push --follow-tags

hooks:
	pre-commit install
	pre-commit install -t pre-push
