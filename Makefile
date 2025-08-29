.PHONY: core api
core:
	$(MAKE) -C packages/plainera_core $(t)

api:
	$(MAKE) -C services/public_api $(t)

# --- Monorepo CI Orchestrator ----------------------------------------------

SUBDIRS := \
	packages/plainera_core \
	packages/plainera_observability \
	services/unacronym_api

STEPS := install lint typecheck test build

.PHONY: ci-local $(STEPS) $(addprefix run-,$(STEPS)) changed run-changed

MAKEFLAGS += --no-print-directory

ci-local:
	@set -e; \
	for d in $(SUBDIRS); do \
	  echo ""; echo "==> $$d: ci-local"; \
	  $(MAKE) -C $$d ci-local; \
	done; \
	echo ""; echo "✅ Monorepo CI local complete"

# Run a single project: make run-ci DIR=services/unacronym_api
run-ci:
	@test -n "$(DIR)" || (echo "Usage: make run-ci DIR=<path>"; exit 2)
	$(MAKE) -C $(DIR) ci-local

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
		if [ -f "$$d/Makefile" ]; then $(MAKE) -C $$d ci-local; else echo "    (no Makefile)"; fi; \
	done

POETRY ?= poetry
PYTHON  ?= python3.13

.PHONY: poetry-config
poetry-config:
	@$(POETRY) --version >/dev/null
	$(POETRY) config virtualenvs.in-project true
	$(POETRY) config keyring.enabled false

.PHONY: check-python
check-python:
	@cmd="$(shell which $(PYTHON))"; \
	if [ -z "$$cmd" ]; then echo "$(PYTHON) not found on PATH"; exit 1; fi; \
	v="$$( $$cmd -c 'import sys;print(".".join(map(str,sys.version_info[:2])))' )"; \
	if [ "$$v" != "3.13" ]; then echo "Expected Python 3.13, got $$v"; exit 1; fi; \
	echo "Using $$cmd (Python $$v)"

.PHONY: bootstrap
bootstrap: check-python poetry-config env-observability env-core env-api

.PHONY: env-observability
env-observability:
	cd packages/plainera_observability && \
	POETRY_KEYRING_ENABLED=0 PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring \
	$(POETRY) env use $(shell which $(PYTHON)) && \
	$(POETRY) sync --no-root --no-interaction -vv

.PHONY: env-core
env-core:
	cd packages/plainera_core && \
	POETRY_KEYRING_ENABLED=0 PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring \
	$(POETRY) env use $(shell which $(PYTHON)) && \
	$(POETRY) sync --no-root --no-interaction -vv

.PHONY: env-api
env-api:
	cd services/unacronym_api && \
	POETRY_KEYRING_ENABLED=0 PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring \
	$(POETRY) env use $(shell which $(PYTHON)) && \
	$(POETRY) sync --no-interaction -vv

.PHONY: lock-all
lock-all:
	$(MAKE) -C packages/plainera_observability lock
	$(MAKE) -C packages/plainera_core lock
	$(MAKE) -C services/unacronym_api lock

.PHONY: deps-check
deps-check:
	$(PYTHON) tools/check_deps.py \
	  packages/plainera_observability/pyproject.toml \
	  packages/plainera_core/pyproject.toml \
	  services/unacronym_api/pyproject.toml

.PHONY: shell-core
shell-core:
	@echo "Run this to activate plainera_core:"
	@echo "  source packages/plainera_core/.venv/bin/activate"


.PHONY: shell-api
shell-api:
	@echo "Run this to activate unacronym_api:"
	@echo "  source services/unacronym_api/.venv/bin/activate"
