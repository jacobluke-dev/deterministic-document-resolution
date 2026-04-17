# document_resolution

## Overview

`document_resolution` is the main resolution package in this repository.

It contains the core document-analysis logic used to identify and resolve:

- acronyms,
- defined terms,
- structural references.

This package holds the main processing logic for those pipelines rather than delivery concerns. Detection, extraction, heuristics, orchestration primitives, and package-level wiring live here, while API transport, request handling, and response shaping sit elsewhere in the repository.

## Purpose

The purpose of this package is to provide a deterministic and reusable resolution engine for document-specific language.

In structured documents, important meaning is often introduced locally and then reused throughout the text. Acronyms, quoted terms, and structural references can all carry document-specific meaning that is easy to lose if processing leans too heavily on general language assumptions.

This package exists to identify and resolve those elements in a way that is explicit, inspectable, and testable.

It is intended to remain reusable outside a single delivery layer, even though the public API service is currently the main integration point in this repository.

## What lives in this package

At a high level, this package contains:

- NLP detection logic,
- extraction flows for acronyms, defined terms, and structural references,
- orchestration contracts and runner registration,
- plugin and activation support,
- package-level composition and wiring.

## Package structure

```text
src/document_resolution/
├── nlp/              # Detection, extraction, heuristics, and pipeline logic
├── orchestration/    # Pipeline contracts, registry, state, and execution helpers
├── wiring/           # Package-level composition and observability wiring
├── db/               # Package-local database-related models
└── __init__.py
````

## Key areas

### `nlp/`

This is the main body of the package.

It contains the document-processing logic for:

* acronym detection and extraction,
* defined-term detection and resolution,
* structural-reference detection and resolution,
* shared heuristics,
* common NLP types and configuration,
* plugin activation and domain-specific gates where needed.

This is where most of the problem-specific behaviour in the repository lives.

### `orchestration/`

This area provides pipeline-level orchestration primitives, including:

* stable pipeline keys,
* request and result contracts,
* pipeline runners,
* registry resolution,
* orchestration state,
* concurrent execution helpers.

The intention is to provide a package-level orchestration seam that is pipeline-aware without being tightly coupled to a specific service or transport layer.

### `wiring/`

This contains package-level composition helpers and observability wiring used to assemble package components cleanly and keep setup concerns separate from the underlying processing logic.

### `db/`

This contains package-local database-related models, including `PackageLogger`.

## How this package is used

In the wider repository, this package is consumed primarily by `services/document_resolution_api`.

The API layer is responsible for:

* request validation,
* dependency injection,
* glossary access,
* response shaping,
* API-specific execution concerns.

This package is responsible for the underlying resolution logic that the API invokes and exposes.

## Design notes

A number of the extraction paths in this package are structured as explicit stage functions operating over shared pipeline state rather than as a single opaque end-to-end flow.

That was a deliberate design choice. It made intermediate outputs easier to inspect, helped trace how a result was formed, made heuristic behaviour easier to debug, and made regressions easier to isolate when a pipeline changed.

The same preference for traceability influenced the use of tiered resolution in parts of the package. Separating earlier deterministic passes from later refinement stages made behaviour easier to reason about and gave me better control over debugging, tuning, and reporting than a flatter pipeline would have.

There are also areas where the package still reflects the path the project took rather than the cleanest final abstraction.

One example is the orchestration boundary. This package contains a generic orchestration layer in `document_resolution.orchestration`, while the public API also contains an API-facing orchestration layer because API execution currently needs chunked execution, timeout handling, and response-oriented control that the lower-level package layer does not model cleanly on its own.

That split is functional, but it also means some orchestration methodology is duplicated across package and service boundaries. I am aware of that trade-off. If I were refining this further, I would likely consolidate that boundary so that responsibility sits more clearly in one place.

A second refinement area is that some pipeline flow modules currently sit under `nlp/extraction`, even where they also coordinate lower-level detection steps. That grouping is practical and keeps related logic close together, but the naming boundary is broader than ideal. In a further cleanup pass, I would likely separate pure extraction logic from higher-level pipeline coordination more explicitly.

## Design principles

This package was built around a few core ideas:

* **Deterministic behaviour where possible**
  Preference is given to defined heuristics and explicit decision paths rather than opaque processing.

* **Stage-based flow design for debugging and testing**
  A number of extraction paths are structured as explicit flows and stages so that intermediate state is easier to inspect, test, and debug.

* **Explicit and inspectable logic**
  Pipeline behaviour should be understandable and traceable rather than hidden behind compressed control flow.

* **Portability across execution contexts**
  Core package logic is intended to remain reusable across packages and delivery layers where possible.

* **Separation between processing and delivery concerns**
  The package is intended to hold document-processing logic, while API and transport-specific concerns sit elsewhere in the repository.

* **Shared pipeline patterns where they add consistency**
  Base pipeline and stage patterns are used in places to keep flow structure more consistent and easier to test.

Those principles are followed more strongly in some parts of the package than in others, but they remain the intended direction of the design.

## Relationship to the rest of the repository

This package is best read alongside:

* `services/document_resolution_api/` for the main service integration,
* `packages/document_resolution_core/` for shared lower-level supporting components,
* `packages/document_resolution_observability/` for logging and observability concerns,
* `packages/document_resolution_rag_demo/` for one downstream application of the package.

## Testing

This package has automated test coverage and is intended to be runnable locally either as part of the wider monorepo test suite or in isolation from the package directory.

```bash
poetry run pytest
```
