FROM python:3.13.3-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir poetry==1.8.3
# only need these for dependency export

COPY services/public_api/pyproject.toml services/public_api/poetry.lock* ./
RUN poetry export -f requirements.txt --without-hashes -o requirements.txt

FROM python:3.13.3-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000

ENV PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN addgroup --system app && adduser --system --ingroup app app
WORKDIR /app

# bring in requirements and local path deps
COPY --from=builder /app/requirements.txt ./
# because build context is repo root (see compose), this path exists:
COPY packages /packages

RUN pip install --no-cache-dir -r requirements.txt

# app code
COPY services/public_api/src ./src
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s CMD curl -fsS http://localhost:${PORT}/healthz || exit 1

USER app
CMD ["uvicorn", "--app-dir", "/app/src", "public_api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
