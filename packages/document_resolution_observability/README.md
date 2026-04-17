# document_resolution_observability

## Overview

`document_resolution_observability` is the shared observability and logging package for this repository.

It provides structured logging, request-level metadata, log context, middleware, redaction, and related observability behaviour across services and packages. Its role is to keep these concerns separate from both the core document-resolution logic and the API delivery layer.

This package is supporting infrastructure and an important part of how the wider system remains inspectable and easier to operate.

## Purpose

The purpose of this package is to provide a reusable home for observability concerns that would otherwise be duplicated or embedded too deeply in service code.

In practice, this includes:

- structured logging support,
- request and context propagation,
- logging decorators and emitters,
- HTTP-level observability helpers,
- redaction and logging hygiene,
- shared observability-facing types.

The intention is to keep operational visibility as a first-class concern rather than an afterthought.

## What lives in this package

At a high level, this package contains:

- logging decorators and emitters,
- request and context helpers,
- middleware for HTTP-related observability,
- body-size and request-id helpers,
- shared observability types,
- lower-level DB-related mixins and model-base support where relevant.

## Package structure

```text
src/observability/
├── config.py            # Package-level observability configuration
├── core/
│   └── types.py         # Shared core types
├── db/
│   ├── mixins.py        # Shared DB-related mixins
│   └── models/
│       └── base.py      # Base database model support
├── http/
│   ├── body_limit.py    # HTTP body-size handling
│   └── request_id.py    # Request ID support
├── logger/
│   ├── access_middleware.py
│   ├── context.py
│   ├── decorator.py
│   ├── emit.py
│   ├── levels.py
│   ├── message_logger.py
│   └── redact.py
└── __init__.py
````

## Key areas

### `logger/`

This is the main body of the package.

It contains the components used for structured logging, context propagation, access logging, message emission, level handling, and redaction. This is where most of the practical observability behaviour lives.

### `http/`

Contains small HTTP-focused helpers for request-level observability concerns, such as request IDs and body-limit support.

### `core/`

Contains shared lower-level observability types used by the rest of the package.

### `db/`

Contains lower-level database-related support used by observability models, such as mixins and base model definitions.

## How this package relates to the rest of the repository

This package is a support layer used by higher-level services and packages across the repository.

It is most relevant to:

* `services/document_resolution_api/`
* `packages/document_resolution/`

Its role is not to drive the document-resolution logic directly, but to make the behaviour of the wider system easier to inspect, trace, and operate.

## Design notes

This package is intentionally focused and infrastructural in nature.

It should not be read as a major domain package, but as shared support code that helps the rest of the repository behave more transparently. The value here is less about algorithmic complexity and more about keeping logging, request metadata, and observability concerns explicit, reusable, and separate from application logic.

Some internal boundaries could still be sharpened. For example, the `logger/` package currently includes middleware and request-context concerns that could arguably sit under `http/` or a more explicitly named context-oriented module. The current layout is functional and keeps related observability behaviour close together, but it also reflects a practical grouping rather than a perfectly final separation of responsibilities.

Overall, the package is small and useful in its current form, though a further refinement pass would likely tighten a few of these internal package boundaries.

## Design principles

This package was built around a few simple ideas:

* keep observability concerns separate from business and delivery logic,
* make structured logging explicit rather than incidental,
* preserve traceability and request-level context where possible,
* treat redaction and logging hygiene as part of the design,
* keep shared operational helpers reusable across services.

## Development

Run tests from this package directory with:

```bash
poetry run pytest
```

Depending on your workflow, broader checks may also be run from the monorepo root.
