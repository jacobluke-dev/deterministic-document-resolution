POETRY ?= poetry
RUN := $(POETRY) run
PKG := plainera_core

.PHONY: install fmt lint style typecheck test cover htmlcov clean ci-local help

help:
	@echo "Common targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sed -e 's/:.*##/: /' | sort

install:
	$(POETRY) install --no-interaction

# Format code with Ruff formatter
fmt:
	$(RUN) ruff format $(PKG) tests

# Lint (and autofix simple issues)
lint:
	$(RUN) ruff check $(PKG) tests --fix

# Optional: CI-friendly style check (no changes)
style:
	$(RUN) ruff format --check $(PKG) tests
	$(RUN) ruff check $(PKG) tests

typecheck:
	$(RUN) mypy $(PKG)

test:
	ENVIRONMENT=TEST $(RUN) pytest -q \
	  --junitxml=pytest-results.xml \
	  --cov=$(PKG) \
	  --cov-report=term \
	  --cov-report=xml:coverage.xml \
	  --cov-fail-under=80

cover:
	$(RUN) pytest --cov=$(PKG) --cov-report=term --cov-report=xml --cov-fail-under=80

htmlcov:
	$(RUN) pytest --cov=$(PKG) --cov-report=html

build: ## Build sdist+wheel
	$(POETRY) build

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml dist build

ci-local: ## Simulate CI locally (lint, typecheck, test, build)
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test
	$(MAKE) build
