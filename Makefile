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
