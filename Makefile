SHELL := /usr/bin/env bash
export PATH := $(HOME)/.local/bin:$(PATH)
POETRY ?= $(shell command -v poetry 2>/dev/null || echo $(HOME)/.local/bin/poetry)
VENV ?= $(shell $(POETRY) env info -p 2>/dev/null)
VENV_BIN ?= $(VENV)/bin
PYTHON ?= python3.13

# Submodules and steps
SUBDIRS := \
    packages/plainera_core \
    packages/plainera_observability \
    services/unacronym_api

STEPS := install lint typecheck test build

# Ensure Poetry environment is configured
MAKEFLAGS += --no-print-directory

ROOT := $(CURDIR)

# Common Help Command
help:
	@echo "Common targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sed -e 's/:.*##/: /' | sort

bootstrap:
	poetry install --with dev
	poetry sync

# Install dependencies for the entire monorepo
install:
	$(POETRY) install --no-interaction

# Check Python version and Poetry configuration
check-python:
	@cmd="$(shell which $(PYTHON))"; \
	if [ -z "$$cmd" ]; then echo "$(PYTHON) not found on PATH"; exit 1; fi; \
	v="$$( $$cmd -c 'import sys;print(".".join(map(str,sys.version_info[:2])))' )"; \
	if [ "$$v" != "3.13" ]; then echo "Expected Python 3.13, got $$v"; exit 1; fi; \
	echo "Using $$cmd (Python $$v)"

poetry-info:
	@$(POETRY) --version

# Format code with Ruff formatter
fmt:
	$(POETRY) run ruff format $(PKG) tests

# Lint (and autofix simple issues)
lint:
	$(POETRY) run ruff check $(PKG) tests --fix

# Style check (no changes)
style:
	$(POETRY) run ruff format --check $(PKG) tests
	$(POETRY) run ruff check $(PKG) tests

# Type checking with mypy
typecheck:
	$(POETRY) run mypy $(PKG)

# Run tests with coverage and junit output
test:
	ENVIRONMENT=TEST $(POETRY) run pytest -q \
	  --junitxml=pytest-results.xml \
	  --cov=$(PKG) \
	  --cov-report=term \
	  --cov-report=xml:coverage.xml \
	  --cov-fail-under=80

# Produce a combined HTML coverage report at repo root using the root venv,
# even if some subprojects' tests fail.
coverage-html: install  ## Combined HTML coverage at htmlcov/index.html
	@rm -f .coverage .coverage.* 2>/dev/null || true
	@root_py="$(VENV_BIN)/python"; \
	if [ ! -x "$$root_py" ]; then echo "Root venv not ready. Run: make install"; exit 1; fi; \
	for d in $(SUBDIRS); do \
	  case "$$d" in \
	    packages/plainera_core)          cov_mod=plainera_core ;; \
	    packages/plainera_observability) cov_mod=plainera_observability ;; \
	    services/unacronym_api)          cov_mod=unacronym_api ;; \
	    *)                                cov_mod=. ;; \
	  esac; \
	  name=$$(basename $$d); \
	  echo ""; echo "==> $$d: collecting coverage with root venv (will continue on failure)"; \
	  (cd $$d && \
	    COVERAGE_FILE=$(ROOT)/.coverage.$$name \
	    "$$root_py" -m pytest -q \
	      --maxfail=1 \
	      --cov="$$cov_mod" \
	      --cov-branch \
	      --cov-report= \
	    ) || true; \
	done; \
	echo ""; echo "==> Combining and rendering HTML"; \
	"$(VENV_BIN)/python" -m coverage combine || true; \
	"$(VENV_BIN)/python" -m coverage html -d htmlcov || true; \
	"$(VENV_BIN)/python" -m coverage report -m || true; \
	echo ""; echo "📄 HTML report: htmlcov/index.html"


# Build sdist and wheel files
build:
	$(POETRY) build

# Clean up caches and build artifacts
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml dist build

# Run all tasks locally like CI (lint, typecheck, test, build)
ci-local: bootstrap ## Simulate CI locally across all packages using root venv
	@set -e; \
	for d in $(SUBDIRS); do \
	  echo ""; echo "==> $$d: ci-local"; \
	  $(MAKE) -C $$d ci-local RUN="$(VENV_BIN)/python -m"; \
	done; \
	echo ""; echo "✅ Monorepo CI local complete"


# Run a single project: make run-ci DIR=services/unacronym_api
run-ci:
	@test -n "$(DIR)" || (echo "Usage: make run-ci DIR=<path>"; exit 2)
	$(MAKE) -C $(DIR) ci-local

# Run a specific step (like lint, typecheck) in all submodules
run-%:
	@set -e; \
	for d in $(SUBDIRS); do \
		echo ""; echo "==> $$d: $*"; \
		if $(MAKE) -C $$d -n $* >/dev/null 2>&1; then \
			$(MAKE) -C $$d $*; \
		else \
			echo "    (skipping: '$*' not defined in $$d)"; \
		fi; \
	done

# Optional: only run changed subdirs (compares to main)
CHANGED_SUBDIRS := $(shell git diff --name-only --relative --merge-base HEAD origin/main | \
	awk -F/ '/^(packages|services)\//{print $$1"/"$$2"/"$$3}' | sort -u)

run-changed:
	@set -e; \
	if [ -z "$(CHANGED_SUBDIRS)" ]; then echo "No changes detected"; exit 0; fi; \
	for d in $(CHANGED_SUBDIRS); do \
		echo ""; echo "==> $$d: ci-local"; \
		if [ -f "$$d/Makefile" ]; then$(MAKE) -C $$d ci-local RUN="$(POETRY) run"; else echo "    (no Makefile)"; fi; \
	done

# Ensure submodules are using Poetry's virtual environment
env-%:
	cd $* && \
	POETRY_KEYRING_ENABLED=0 PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring \
	$(POETRY) env use $(shell which $(PYTHON)) && \
	$(POETRY) sync --no-interaction -vv

# Lock all submodules
lock-all:
	$(MAKE) -C packages/plainera_observability lock
	$(MAKE) -C packages/plainera_core lock
	$(MAKE) -C services/unacronym_api lock

# Check dependencies across all submodules
deps-check:
	$(PYTHON) tools/check_deps.py \
	  packages/plainera_observability/pyproject.toml \
	  packages/plainera_core/pyproject.toml \
	  services/unacronym_api/pyproject.toml
