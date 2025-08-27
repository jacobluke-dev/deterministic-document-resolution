POETRY=poetry
RUN=$(POETRY) run

.PHONY: install fmt lint typecheck test style cover htmlcov clean hooks build

install:
	$(POETRY) install

# Format code with Ruff formatter
fmt:
	$(RUN) ruff format .

# Lint (and autofix simple issues)
lint:
	$(RUN) ruff check . --fix

# Optional: CI-friendly style check (no changes)
style:
	$(RUN) ruff format --check .
	$(RUN) ruff check .

typecheck:
	$(RUN) mypy --config-file pyproject.toml --package plainera_core

test:
	$(RUN) pytest -q --cov=plainera_core --cov-report=term --cov-report=xml --cov-fail-under=80

cover:  # just re-run coverage without -q
	$(RUN) pytest --cov=plainera_core --cov-report=term --cov-report=xml --cov-fail-under=80

htmlcov:  # open HTML coverage locally
	$(RUN) pytest --cov=plainera_core --cov-report=html
	@python -c "import webbrowser; webbrowser.open('htmlcov/index.html')" >/dev/null 2>&1 || true

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build

hooks:
	pre-commit install
	pre-commit install -t pre-push

build:
	$(POETRY) build
