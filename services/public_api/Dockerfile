FROM python:3.13.3-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir poetry==1.8.3

COPY pyproject.toml poetry.lock* ./
RUN poetry export -f requirements.txt --without-hashes -o requirements.txt

FROM python:3.13.3-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN addgroup --system app && adduser --system --ingroup app app
WORKDIR /app

COPY --from=builder /app/requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY .env ./.env

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s CMD curl -fsS http://localhost:${PORT}/healthz || exit 1

USER app
CMD ["uvicorn", "public_api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
