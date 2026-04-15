
# Document Resolution

## Overview

Document Resolution is a public-facing monorepo for deterministic analysis of structured documents.

The core purpose of the project is to identify and resolve document-specific language in a traceable and reusable way, with a particular focus on:

- acronyms,
- defined terms,
- structural references.

The repository includes the core resolution engine, a public API service, a web application, infrastructure definitions, and a RAG demonstrator used to explore grounded versus baseline retrieval behaviour.

This project also serves a second purpose. It is a practical demonstration of my ability to take a broad problem, break it down into meaningful technical concerns, architect a modular solution, and implement it across packages, services, infrastructure, and application layers. Although this repo has been primarily designed and developed by me, it has been structured so that it could be understood, extended, and worked on collaboratively.

## TL;DR version

- Deterministic document-resolution monorepo focused on acronyms, defined terms, and structural references.
- Includes reusable Python packages, a FastAPI service, a Next.js web app, infrastructure definitions, and a grounded-vs-baseline RAG demonstrator.
- Built both to explore a real document-processing problem and to demonstrate end-to-end engineering across architecture, packages, services, and delivery structure.
- Functional and actively usable, with a few known refinement areas documented below.

## Why this project exists

A large amount of important document meaning sits in local definitions rather than in general language.

In real documents, a term such as an acronym or quoted phrase often has a specific meaning that is introduced once and then reused throughout the rest of the text. If that meaning is not preserved, downstream systems can drift, misinterpret the document, or produce plausible but incorrect outputs.

This repository exists to explore and implement a more deterministic approach to that problem.

It was built to answer two related questions:

1. How can document-specific meanings be extracted and resolved in a structured, testable, and reusable way?
2. How can a larger engineering problem be taken from concept through design, decomposition, implementation, and operational structure in a way that is clear and maintainable?

## What this repository contains

This monorepo contains:

- reusable Python packages for document resolution, domain logic, observability, and demo pipelines,
- a FastAPI service exposing public-facing endpoints,
- a Next.js web application for interacting with the system,
- Terraform infrastructure definitions,
- Docker and local development tooling,
- supporting test and dependency-checking utilities.

This repository is a public-facing version of the work. Private or internal-only project elements have been removed so that the repo can be shared openly.

## Architecture overview

At a high level, the repository is split into separate concerns:

- **apps/** contains user-facing applications, (work in progress)
- **packages/** contains reusable libraries and domain logic,
- **services/** contains deployable back-end services,
- **infra/** contains infrastructure definitions, `TODO`
- **tools/** contains repository maintenance and utility scripts.

This separation is intentional. The goal is to keep domain logic reusable, keep deployment concerns isolated from library code, and make it easier to reason about where responsibilities sit across the system.

## Repository structure

```text
.
├── apps/
│   └── documentResolutionWeb/          # Next.js web application (work in progress)
├── infra/
│   └── terraform/                      # Infrastructure definitions TODO
├── packages/
│   ├── document_resolution/            # Main resolution engine
│   ├── document_resolution_core/       # Core domain and shared services
│   ├── document_resolution_observability/  # Logging and observability
│   ├── document_resolution_rag_demo/   # Grounded vs baseline RAG demonstrator
│   └── document_resolution_testkit/    # Test helpers and fixtures
├── services/
│   └── document_resolution_api/        # FastAPI public API service
├── tools/
│   └── check_deps.py                   # Dependency consistency checks
├── docker-compose.yml
├── pyproject.toml
├── package.json
└── README.md
````

## Key components

### `apps/documentResolutionWeb`

A Next.js application that provides a user-facing interface to the system. It includes the front-end application shell, UI components, and API route integration for resolution workflows. This is still a work in progress and is not yet connected

### `packages/document_resolution`

The main resolution package. This contains the bulk of the NLP, heuristics and extraction logic for acronyms, defined terms, structural references, orchestration, and wiring. With abstraction and expansion abilities for example a date/time pipeline.

### `packages/document_resolution_core`

Core shared domain concepts and lower-level service components, such as database management, sinks and factories. Elements of this package require rework see [design notes](#design-notes-and-known-refinement-areas).

### `packages/document_resolution_observability`

Observability and logging support, including middleware, request handling utilities, logger infrastructure, and related shared model types.

### `packages/document_resolution_rag_demo`

A demonstrator package used to explore grounded retrieval and compare it against baseline RAG behaviour. This sits alongside the main resolution engine rather than inside it because it is an application of the core system rather than the core system itself.

### `packages/document_resolution_testkit`

Shared test utilities, fixtures, and support helpers for package and service-level testing.

### `services/document_resolution_api`

A FastAPI service exposing public endpoints for health checks, resolution requests, orchestration, and demo-related functionality. This is the primary deployable back-end entry point for the repository.

### `infra/terraform`

Infrastructure definitions for provisioning supporting components, such as PostgreSQL-related infrastructure and outputs. contains infrastructure definitions; this area is present but not yet active in the current local workflow..

### `tools`

Repository-level scripts for housekeeping and consistency checks. For example, `check_deps.py` helps verify dependency version alignment across packages and services.

## Design notes and known refinement areas

This repository is functional and structured deliberately, but not every part of it is equally mature.

Some internal abstractions reflect earlier design iterations and would be simplified if I were continuing an active refinement pass. For example, parts of `document_resolution_core` are intentionally lightweight and currently provide less architectural value than I would want in a more polished version of the system. In practice, this does not block the system from running correctly, but it is an area where I would likely reduce indirection, remove weaker abstractions, and tighten package responsibilities.

A similar point applies to orchestration. The codebase currently has a generic orchestration layer in `document_resolution.orchestration` and a second API-facing orchestration layer in `public_api.core.orchestration`. That split is functional, but it also reflects an earlier design direction: the lower-level layer handles runner registration, target resolution, and generic state accumulation, while the API layer still owns chunked execution, timeout handling, and response-oriented execution concerns.

In practice, this means some orchestration control flow is duplicated across layers. I am aware of that trade-off. If I were continuing an active refinement pass, I would likely consolidate those responsibilities so the architectural boundary is clearer. The likely end state would be either a genuinely reusable orchestration engine in `document_resolution` with a thin API adapter, or a deliberately lightweight runner layer in `document_resolution` with the API as the true orchestrator.

## Technology stack

This repository uses:

* **Python 3.13**
* **Poetry** - on reflection I would like to move to `uv` after discovering this.
* **FastAPI**
* **Pytest**
* **mypy**
* **Ruff**
* **TypeScript**
* **Next.js**
* **pnpm**
* **Docker / Docker Compose**
* **Terraform** TODO, not yet part of the active deployment workflow
* **PostgreSQL**

Some parts of the demonstrator also use retrieval and embedding-related components where appropriate.

## Design principles

A few principles shaped how this repository was put together:

* **Deterministic behaviour where possible**
  Preference is given to logic that is explicit, inspectable, and testable.

* **Clear separation of concerns**
  Reusable packages, deployable services, applications, and infrastructure are kept distinct.

* **Traceability over magic**
  The intention is that behaviour can be reasoned about rather than guessed at.

* **Extensibility**
  Even where the original implementation was primarily solo, the structure aims to support future collaboration and extension.

* **Operational awareness**
  Logging, observability, environment handling, and deployment structure are treated as part of the system, not afterthoughts.

* **Agnostic design**
  Where possible, the code is written to be reusable across packages and execution contexts rather than being tightly coupled to a single service, interface, or deployment path. The aim is to keep core logic portable and composable, although there are places in the repository where that standard has not been met fully and where responsibilities could be separated more cleanly.

## Getting started

### Prerequisites

You will need:

* Python 3.13
* Poetry
* Node.js
* pnpm
* Docker
* Docker Compose

Depending on what you want to run locally, you may also need environment variables for services such as the API, database, and any external providers used by the demo components.

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <repository-name>
```

### 2. Install Python dependencies

At the repository root:

```bash
poetry install
```

Some packages and services may also be managed independently through their own `pyproject.toml` files depending on how you prefer to work locally.

### 3. Install JavaScript dependencies

At the repository root:

```bash
pnpm install
```

### 4. Configure environment variables

Create the required local environment files for the services or apps you want to run.

Typical examples include:

* API configuration
* database connection settings
* front-end API base URL
* any provider keys required by demo components

Refer to package- or service-specific configuration files and READMEs where present.

## Running the system

### Run with Docker Compose

For a full local environment, use Docker Compose from the repository root:

```bash
docker compose up --build
```

If you use the override file during development:

```bash
docker compose up --build
```

This is the simplest way to bring up the main local dependencies together.


Once the containers are running successfully, at the repository root:

```bash
make ci-local
```

This closely mimics the GitLab CI process if this is successful then generally Gitlab, although tthere may still be environment specific failures.

### Run the API service

From the API service directory:

```bash
cd services/document_resolution_api
poetry install
poetry run uvicorn src.public_api.main:app --reload
```

### Run the web app

From the web app directory:

```bash
cd apps/documentResolutionWeb
pnpm install
pnpm dev
```

## Local development workflow

The repository is organised so that you can work either from the monorepo root or directly within an individual package or service, depending on the task.

Typical workflow:

1. start required local dependencies,
2. run the API service,
3. run the web application,
4. run tests and type checks for the area you are changing,
5. verify dependency consistency where relevant.

## Testing, linting, and type checking

Examples of useful commands include the following.

### Python tests

From the relevant package or service directory:

```bash
poetry run pytest
```

### Type checking

```bash
poetry run mypy .
```

### Linting

If Ruff is configured for the current package or service:

```bash
poetry run ruff check .
```

### Dependency consistency checks

From the repository root:

```bash
python tools/check_deps.py \
  packages/document_resolution/pyproject.toml \
  packages/document_resolution_core/pyproject.toml \
  packages/document_resolution_observability/pyproject.toml \
  packages/document_resolution_rag_demo/pyproject.toml \
  services/document_resolution_api/pyproject.toml
```

Adjust the list of files as needed for the set of packages you want to compare.

## How to read this repository

If you are new to the codebase, the most useful reading order is usually:

1. this top-level `README.md`,
2. `services/document_resolution_api/`,
3. `services/document_resolution_api/migrations/`,
4. `packages/document_resolution/`,
5. `apps/documentResolutionWeb/`,
6. `packages/document_resolution_rag_demo/`,
7. `infra/terraform/`.

That order gives a sensible path from entry points, to core logic, to UI and infrastructure.

## Current status

This is an active, public-facing engineering repository rather than a finished commercial product.

The codebase is intended to be useful in its own right, but it also reflects a broader objective: demonstrating how I approach architecture, package boundaries, service design, delivery structure, and technical problem decomposition across a non-trivial system.

Some elements were removed before publication to make the repository appropriate for public sharing.

## What I wanted this project to demonstrate

Beyond the immediate document-resolution problem, this project was designed to demonstrate that I can:

* take an ambiguous or domain-heavy problem and shape it into a workable technical scope,
* decompose a larger system into coherent packages, services, applications, and infrastructure,
* make architectural decisions that support maintainability and extension,
* build with testing, observability, and operational structure in mind,
* develop a codebase that could realistically be handed to or extended by other engineers.

In that sense, this repository is both a software project and a record of how I think about engineering work.

## Future work

Areas that could be expanded further include:

* broader public documentation,
* deeper deployment guidance,
* richer demo scenarios,
* more complete environment examples,
* additional hardening and operational polish.
* clearer consolidation of orchestration responsibilities across package and API layers.

## Licence

This repository is licensed under the terms set out in the [LICENSE](./LICENSE) file.
