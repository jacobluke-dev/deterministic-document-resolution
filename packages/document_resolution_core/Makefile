SHELL := /usr/bin/env bash
export PATH := $(HOME)/.local/bin:$(PATH)
RUN ?= python -m
PKG := document_resolution_core
SRC := src
TESTS := tests
COV_FAIL_UNDER ?= 80

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
	$(RUN) mypy -p $(PKG)

test:
	# keep coverage by import name (not src path)
	$(RUN) pytest -q --override-ini="addopts=-q --cov=$(PKG) --cov-report=term --cov-report=xml:coverage.xml --cov-fail-under=$(COV_FAIL_UNDER)" tests


cover:
	$(RUN) pytest --cov=$(SRC) --cov-report=term --cov-report=xml --cov-fail-under=$(COVER_FAIL_UNDER)

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
