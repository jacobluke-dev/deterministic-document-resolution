FROM python:3.13.3-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Poetry + export plugin
RUN pip install poetry==1.8.3

# --- make path-deps visible to Poetry during export ---
# copy service metadata and the monorepo packages tree
COPY services/unacronym_api/pyproject.toml services/unacronym_api/poetry.lock* ./services/unacronym_api/
COPY packages ./packages

# export runtime (non-dev) requirements to a file *at /app*
WORKDIR /app/services/unacronym_api
RUN poetry export -f requirements.txt --without dev --without-hashes -o /app/requirements.txt
# ------------------------------------------------------

FROM python:3.13.3-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000
ENV PYTHONPATH=/app/src:/app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN addgroup --system app && adduser --system --ingroup app app
WORKDIR /app

# 1) requirements from builder
COPY --from=builder /app/requirements.txt ./

# 2) copy packages to **/app/packages** to match file:///app/packages/... in requirements.txt
COPY packages /app/packages

# 3) install requirements
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
         -e /app/packages/plainera_core \
         -e /app/packages/plainera_observability[fastapi] \
         -e /app/packages/plainera_unacronym

# 4) app code
COPY services/unacronym_api/src ./src

EXPOSE 8000
HEALTHCHECK --interval=120s --timeout=3s --start-period=10s CMD curl -fsS http://localhost:${PORT}/healthz || exit 1
USER app
CMD ["uvicorn", "--app-dir", "/app/src", "public_api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
