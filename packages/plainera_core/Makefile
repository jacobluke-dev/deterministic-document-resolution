SHELL := /usr/bin/env bash
export PATH := $(HOME)/.local/bin:$(PATH)
RUN ?= python -m
SRC := src
TESTS := tests

.PHONY: install fmt lint style typecheck test cover htmlcov clean ci-local help

help:
	@echo "Common targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sed -e 's/:.*##/: /' | sort

# Format code with Ruff formatter
fmt:
	$(RUN) ruff format $(SRC) $(TESTS)

# Lint (and autofix simple issues)
lint:
	$(RUN) ruff check $(SRC) $(TESTS) --fix

# Optional: CI-friendly style check (no changes)
style:
	$(RUN) ruff format --check $(SRC) $(TESTS)
	$(RUN) ruff check $(SRC) $(TESTS)

typecheck:
	ROOT=$$(git rev-parse --show-toplevel); \
	PYTHONPATH=$$ROOT/services/unacronym_api/src:src $(RUN) mypy $(SRC)

test:
	@set -e; \
	ROOT=$$(git rev-parse --show-toplevel); \
	PYTHONPATH="$$ROOT/services/unacronym_api/src:src" $(RUN) pytest -q \
	  --override-ini="addopts=-q --cov=src --cov-report=term --cov-report=xml:coverage.xml --cov-fail-under=80" \
	  tests

cover:
	$(RUN) pytest --cov=$(SRC) --cov-report=term --cov-report=xml --cov-fail-under=80

htmlcov:
	$(RUN) pytest --cov=$(SRC) --cov-report=html

build: ## Build sdist+wheel
	$(RUN) build .

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml dist build

ci-local: ## Simulate CI locally (lint, typecheck, test, build)
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test
	$(MAKE) build
