# services/public_api/Makefile
POETRY ?= poetry
RUN := $(POETRY) run
PORT ?= 8000
IMAGE ?= unacronym-api
GIT_SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)

.PHONY: install run fmt lint style typecheck test cover htmlcov clean hooks build up down help


help:
	@echo "Common targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sed -e 's/:.*##/: /' | sort



install:
	$(POETRY) install --no-interaction

# Dev server

# ---- Dev & Quality ---------------------------------------------
run: ## Run app locally with reload
	PYTHONPATH=src $(RUN) uvicorn public_api.main:create_app --factory --reload --host 0.0.0.0 --port $(PORT)

# Format with Ruff formatter
fmt:
	$(RUN) ruff format src tests

# Lint (and autofix simple issues)
lint:
	$(RUN) ruff check src tests --fix

# CI-friendly style check (no writes)
style:
	$(RUN) ruff format --check src tests
	$(RUN) ruff check src tests

# Type checking (uses local mypy.ini or pyproject)
typecheck:
	$(RUN) mypy --config-file mypy.ini

# Tests + coverage
test:
	PYTHONPATH=src $(RUN) pytest -q \
	  --junitxml=pytest-results.xml \
	  --cov=public_api \
	  --cov-report=term \
	  --cov-report=xml:coverage.xml \
	  --cov-fail-under=80

cover:  # verbose coverage
	$(RUN) pytest --cov=public_api --cov-report=term --cov-report=xml --cov-fail-under=80

htmlcov:
	$(RUN) pytest --cov=public_api --cov-report=html
	@python -c "import webbrowser; webbrowser.open('htmlcov/index.html')" >/dev/null 2>&1 || true

# ---- Migrations (Alembic) --------------------------------------
ALEMBIC?=alembic
alembic-help:
	@$(ALEMBIC) -h || (echo "Alembic not installed. Add it to dev deps and alembic.ini." && exit 1)

migrate: alembic-help ## Apply migrations (upgrade head)
	$(ALEMBIC) upgrade head

downgrade: alembic-help ## Roll back one migration
	$(ALEMBIC) downgrade -1

# ---- Seed -------------------------------------------------------
seed-local: ## Seed sample data (only if APP_ENV=local)
	@if [ "$$APP_ENV" != "local" ]; then echo "Refusing to seed unless APP_ENV=local"; exit 2; fi
	@echo "Seeding sample glossary entries... (stub)"; exit 0

# ---- Build & CI -------------------------------------------------
IMAGE ?= unacronym-api
APP_VERSION ?= 0.1.0
LABELS = --label org.opencontainers.image.title=$(IMAGE) \
         --label org.opencontainers.image.version=$(APP_VERSION) \
         --label org.opencontainers.image.revision=$(GIT_SHA)

build: ## Docker build with labels
	docker build $(LABELS) -t $(IMAGE):$(GIT_SHA) -t $(IMAGE):latest .

# Optional: local compose (only API for now)
up:
	docker compose up --build -d

down:
	docker compose down

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
