# document_resolution

## Overview

`document_resolution` is the main resolution package in this repository.

It contains the bulk of the document-analysis logic used to identify and resolve:

- acronyms,
- defined terms,
- structural references.

This package is intended to hold the core pipeline logic rather than deployment concerns. In practice, that means detection, extraction, orchestration primitives, pipeline wiring, and related NLP-focused behaviour live here,
while API delivery and request-specific composition sit elsewhere in the repository.

## Purpose

The purpose of this package is to provide a deterministic and reusable resolution engine for document-specific language.

A large amount of important meaning in structured documents is not carried by general vocabulary alone. Acronyms, quoted terms, and structural references are often defined locally and then reused throughout the document. This package exists to identify those elements in a way that is explicit, inspectable, and testable.

It is designed to be reusable outside a single delivery layer, even though the public API service is currently the main integration point in this repository.

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

It contains the actual document-processing logic, including:

* acronym detection and extraction,
* defined-term detection and resolution,
* structural-reference detection and resolution,
* shared heuristics,
* common NLP types and configuration,
* plugin activation and domain-specific gates where needed.

This is the part of the package that does the real work.

### `orchestration/`

This area provides pipeline-level orchestration primitives:

* stable pipeline keys,
* request and result contracts,
* pipeline runners,
* registry resolution,
* orchestration state,
* concurrent execution helpers.

The intention here is to provide a package-level orchestration seam that is pipeline-aware but not tied directly to a single transport or service layer.

### `wiring/`

This contains package-level composition helpers and observability wiring, for the db sink. It exists to keep assembly concerns separate from the underlying domain logic.

### `db/`

It includes package-local database-related models but is not the main persistence boundary for the wider repository, `PackageLogger` for specific logging in this package.

## How this package is used

In the wider repository, this package is consumed primarily by the public API service under `services/document_resolution_api`.

The API layer is responsible for:

* request validation,
* dependency injection,
* glossary access,
* response shaping,
* API-specific execution concerns.

This package is responsible for the underlying resolution logic that the API orchestrates and exposes.

## Design notes

This package is functional and central to the repository, but not every part of it is equally mature.

Some pipeline flow modules currently sit under `nlp/extraction`, even where they also coordinate lower-level detection steps. That grouping is practical and keeps related logic close together, but it also means the package boundary is somewhat broader than the name alone suggests.

If I were refining this further, I would likely separate pure extraction logic from higher-level pipeline coordination more explicitly.

### Strengths

The strongest parts of this package are the pipeline and NLP flows themselves, particularly the detection, extraction, and heuristic stages. That is where most of the problem-specific logic lives, and it is the area the rest of the system depends on most directly.

One design decision I was deliberate about was breaking a number of the flows into explicit stage functions operating over shared pipeline state, rather than hiding all processing inside all processing inside a single opaque end-to-end flow. That approach made it easier to inspect intermediate outputs, trace how a result was formed, debug heuristic behaviour, and isolate changes when a pipeline regressed.

That same preference for traceability also influenced the use of tiered resolution in parts of the package. Separating earlier deterministic passes from later refinement stages made the behaviour easier to reason about and gave me more control over debugging, tuning, and reporting than a flatter pipeline would have.

### Known refinement areas

Some areas reflect earlier design iterations and would be simplified in a future refinement pass.

One example is the orchestration boundary. This package contains a generic orchestration layer in `document_resolution.orchestration`, but the public API also contains its own orchestration layer because API execution currently needs chunked execution, timeout handling, and response-oriented control that the lower-level package layer does not model cleanly on its own.

As a result, some orchestration methodology is duplicated across package and service boundaries. This is a known design trade-off rather than an unknown issue. If I were continuing an active cleanup pass, I would likely consolidate that boundary so the responsibility sits more clearly in one place.

A second refinement area is that some internal abstractions are lighter than I would want in a more polished final design. In a few places, structure remains because of the path the project took rather than because it is the strongest possible abstraction. The package works correctly, but there are parts I would reduce or simplify in a future pass.

## Design principles

This package was built around a few core ideas:

* **Deterministic behaviour where possible**
  Preference is given to defined heuristics and explicit decision paths rather than opaque processing.

* **Stage-based flow design for debugging and testing**
  A number of extraction paths are structured as explicit flows and stages so that intermediate state is easier to inspect, test, and debug. See, for example, `packages/document_resolution/src/document_resolution/nlp/extraction/structural/extract_flow.py`.

* **Explicit and inspectable logic**
  The intention is that pipeline behaviour can be reasoned about and traced rather than guessed at.

* **Portability across execution contexts**
  Core package logic is intended to remain reusable across packages and delivery layers where possible.

* **Separation between processing and delivery concerns**
  The package is intended to hold document-processing logic, while API and transport-specific concerns sit elsewhere in the repository.

* **Shared pipeline patterns where they add consistency**
  Base pipeline and stage patterns are used in places to keep flow structure more consistent and easier to test. See `packages/document_resolution/tests/test_document_resolution/test_nlp/extraction/base`.

Those principles are followed more strongly in some parts of the package than in others, but they remain the intended direction.


## Relationship to the rest of the repository

This package should usually be read alongside:

* `services/document_resolution_api/` for the main service integration,
* `packages/document_resolution_core/` for shared lower-level supporting components,
* `packages/document_resolution_observability/` for logging and observability concerns,
* `packages/document_resolution_rag_demo/` for one downstream application of the package.

## Current status

This is an active package within a wider public-facing repository.

It is not presented as a perfectly finalised library. Instead, it is intended to show a substantial working resolution engine, the architectural direction behind it, and the places where I believe the design is strong versus the places I would refine further.

## Testing

This package has automated test coverage and is intended to be runnable locally as part of the wider monorepo test suite or in isolation from the package directory.

Run tests from this package with:

```bash
poetry run pytest
```
