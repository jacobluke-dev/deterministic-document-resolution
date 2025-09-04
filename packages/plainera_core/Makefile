SHELL := /usr/bin/env bash
export PATH := $(HOME)/.local/bin:$(PATH)
RUN ?= python -m
PKG := core utils db_manager

.PHONY: install fmt lint style typecheck test cover htmlcov clean ci-local help

help:
	@echo "Common targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sed -e 's/:.*##/: /' | sort

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
	$(RUN) pytest -q --override-ini="addopts=-q --cov=$(PKG) --cov-report=term --cov-report=xml:coverage.xml --cov-fail-under=80" tests

cover:
	$(RUN) pytest --cov=$(PKG) --cov-report=term --cov-report=xml --cov-fail-under=80

htmlcov:
	$(RUN) pytest --cov=$(PKG) --cov-report=html

build: ## Build sdist+wheel
	$(RUN) build .

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml dist build

ci-local: ## Simulate CI locally (lint, typecheck, test, build)
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test
	$(MAKE) build
