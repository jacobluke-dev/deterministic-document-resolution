SHELL := /usr/bin/env bash
POETRY ?= $(shell command -v poetry 2>/dev/null || echo $(HOME)/.local/bin/poetry)
RUN ?= $(POETRY) run
PKG := plainera_observability

.PHONY: ensure-venv install fmt lint style typecheck test build ci-local

ensure-venv:
	@$(POETRY) env info --path >/dev/null 2>&1 || $(POETRY) env use 3.13
	@$(POETRY) run python -c "import ruff,pytest,mypy" >/dev/null 2>&1 || \
	    $(POETRY) install --with dev --sync --dry-run

.PHONY: shell-observability
shell-observability:
	@echo "Run this to activate plainera_observability:"
	@echo "source .venv/bin/activate"

poetry-info:
	@$(POETRY) --version

install:  ## Force (re)install (optional explicit target)
	$(POETRY) install  --with --sync --dry-run dev

lock:
	$(POETRY) lock --no-update

fmt: ensure-venv
	$(RUN) ruff format $(PKG) tests

lint: ensure-venv
	$(RUN) ruff check $(PKG) tests --fix

style: ensure-venv
	$(RUN) ruff format --check $(PKG) tests
	$(RUN) ruff check $(PKG) tests

typecheck: ensure-venv
	$(RUN) mypy $(PKG)

test: ensure-venv
	ENVIRONMENT=TEST PYTHONPATH=. $(RUN) pytest -q \
	  --junitxml=pytest-results.xml \
	  --cov=$(PKG) \
	  --cov-report=term \
	  --cov-report=xml:coverage.xml \
	  --cov-fail-under=80

build: ensure-venv
	$(POETRY) build

ci-local: lint typecheck test build  ## No explicit install needed
	@echo "✅ $(PKG) OK"
