# services/public_api/Makefile
POETRY ?= poetry
RUN := $(POETRY) run
PORT ?= 8000
IMAGE ?= unacronym-api
GIT_SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)

.PHONY: install run fmt lint style typecheck test cover htmlcov clean hooks build up down

install:
	$(POETRY) install --no-interaction

# Dev server
run:
	$(RUN) uvicorn public_api.main:create_app --factory --host 0.0.0.0 --port $(PORT)

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
	$(RUN) pytest -q --cov=public_api --cov-report=term --cov-report=xml --cov-fail-under=80

cover:  # verbose coverage
	$(RUN) pytest --cov=public_api --cov-report=term --cov-report=xml --cov-fail-under=80

htmlcov:
	$(RUN) pytest --cov=public_api --cov-report=html
	@python -c "import webbrowser; webbrowser.open('htmlcov/index.html')" >/dev/null 2>&1 || true

# Docker image
build:
	docker build -t $(IMAGE):$(GIT_SHA) -t $(IMAGE):latest .

# Optional: local compose (only API for now)
up:
	docker compose up --build -d

down:
	docker compose down

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build
	find src -name '__pycache__' -type d -prune -exec rm -rf {} +

hooks:
	pre-commit install
	pre-commit install -t pre-push
