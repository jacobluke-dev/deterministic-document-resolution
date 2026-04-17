# document_resolution_api

## Overview

`document_resolution_api` is the main public service layer for this repository.

It exposes the document-resolution engine over HTTP using FastAPI and acts as the primary deployable back-end entry point for the wider system. In practice, this service is responsible for receiving requests, validating input, coordinating pipeline execution, applying API-specific orchestration concerns, and shaping public responses.

The service sits on top of the reusable packages in `packages/`, particularly `document_resolution`, and is where repository-level document analysis becomes a callable API.

## Purpose

The purpose of this service is to provide a clean public interface over the underlying document-resolution engine.

It is responsible for turning a package-level resolution system into an application-facing service by handling concerns such as:

- request validation,
- dependency injection,
- orchestration entry points,
- timeout and error handling,
- glossary and persistence integration where required,
- response composition.

The intention is to keep delivery concerns here rather than inside the core resolution package.

## What this service contains

At a high level, this service contains:

- FastAPI route definitions,
- dependency injection and service wiring,
- request orchestration and execution control,
- response mapping and schema definitions,
- persistence models and repositories,
- migrations and seed support,
- service-level configuration.

## Service structure

```text
services/document_resolution_api/
├── migrations/      # Alembic migrations and migration support
├── src/
│   ├── public_api/
│   │   ├── api/         # Routers and response-facing API modules
│   │   ├── cli/         # CLI support for API keys
│   │   ├── core/        # Service orchestration, DI, settings, and API-layer services
│   │   ├── db/          # Database models, repositories, migration status, and seed support
│   │   ├── schemas/     # Request/response and shared API schemas
│   │   └── main.py      # FastAPI application entry point
│   └── wiring/      # Service composition helpers
├── alembic.ini
└── README.md
````

## Key areas

### `api/`

Contains the public route layer, including routers, exception handlers, and response-facing API definitions.

### `core/`

Contains the main application logic for the service, including dependency injection, orchestration, request building, mapping, API-layer services, and settings.

This is where package-level resolution logic is adapted into API behaviour.

### `db/`

Contains persistence-related models, repositories, migration helpers, and seed support used by the service.

### `schemas/`

Contains the request and response schemas used by the public API, along with shared API-facing types.

### `main.py`

The FastAPI application entry point.

## How this service relates to the rest of the repository

This service depends primarily on:

* `packages/document_resolution/` for the core document-resolution logic,
* `packages/document_resolution_core/` for lower-level supporting components,
* `packages/document_resolution_observability/` for logging and observability support.

It is also the main back-end integration point for the wider repository, including the web application and demo-oriented flows where applicable.

## Configuration

The service uses environment-based configuration for runtime behaviour such as database access, request limits, timeout settings, logging, and optional integrations.

In practice, the easiest way to understand the required configuration is to review:

* the service settings module,
* the local environment files used by the repository,
* any example environment configuration provided alongside the service.

The full environment variable surface is intentionally not duplicated in this README, as it is better maintained closer to the configuration itself.

## Design notes

This service is one of the stronger architectural boundaries in the repository because it separates application delivery concerns from the underlying resolution engine.

The clearest refinement area is orchestration. The service contains its own API-facing orchestration layer in `public_api.core.orchestration`, while the underlying `document_resolution` package also contains orchestration primitives. That split is functional, but it also reflects an architectural seam that could be clearer. The service-level orchestrator currently owns API-specific concerns such as chunked execution, timeout mapping, and response-oriented execution behaviour that the lower-level orchestration layer does not model cleanly on its own.

As a result, some orchestration control flow is duplicated across package and service boundaries. This is a known refinement area rather than an unknown issue.

## Design principles

This service was built around a few core ideas:

* keep delivery concerns separate from core processing logic,
* expose stable and inspectable request/response behaviour,
* keep orchestration deterministic where possible,
* make service wiring and dependency boundaries explicit,
* support local development and testing without hiding too much behind framework magic.

## Testing

Run tests for this service from this directory with:

```bash
poetry run pytest
```

Depending on the target, there may also be migration, database-backed, or integration-oriented test flows available through local Make targets.

