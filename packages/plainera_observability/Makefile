POETRY ?= poetry
RUN := $(POETRY) run
PKG := plainera_observability

.PHONY: install fmt lint style typecheck test cover htmlcov clean ci-local help

help:
	@echo "Common targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sed -e 's/:.*##/: /' | sort

install: ## Install deps for this package
	$(POETRY) install --no-interaction

fmt: ## Format with Ruff
	$(RUN) ruff format $(PKG) tests

lint: ## Lint (and autofix)
	$(RUN) ruff check $(PKG) tests --fix

style: ## Lint (no writes)
	$(RUN) ruff format --check $(PKG) tests
	$(RUN) ruff check $(PKG) tests

typecheck: ## mypy on the package
	$(RUN) mypy $(PKG)

test: ## Tests + coverage
	ENVIRONMENT=TEST $(RUN) pytest -q \
	  --junitxml=pytest-results.xml \
	  --cov=$(PKG) \
	  --cov-report=term \
	  --cov-report=xml:coverage.xml \
	  --cov-fail-under=80

build: ## Build sdist+wheel
	$(POETRY) build

cover:
	$(RUN) pytest --cov=$(PKG) --cov-report=term --cov-report=xml --cov-fail-under=80

htmlcov:
	$(RUN) pytest --cov=$(PKG) --cov-report=html

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml dist build

ci-local: ## Simulate CI locally (lint, typecheck, test, build)
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test
	$(MAKE) build
