SHELL := /usr/bin/env bash
RUN ?= python -m

PKG := observability
FORMAT_PATHS := $(PKG)
LINT_PATHS := $(PKG) tests

.PHONY: fmt lint style typecheck test build ci-local

fmt:
	$(RUN) ruff format $(FORMAT_PATHS)

lint:
	$(RUN) ruff check $(LINT_PATHS) --fix

style:
	$(RUN) ruff format --check $(FORMAT_PATHS)
	$(RUN) ruff check $(LINT_PATHS)

typecheck:
	$(RUN) mypy $(PKG)

test:
	$(RUN) pytest -q --override-ini="addopts=-q --cov=$(PKG) --cov-report=term --cov-report=xml:coverage.xml --cov-fail-under=80" tests

build:
	$(RUN) build .

ci-local: fmt lint typecheck test build
	@echo "✅ $(PKG) OK"
