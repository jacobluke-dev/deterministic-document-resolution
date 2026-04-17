# document_resolution_rag_demo

## Overview

`document_resolution_rag_demo` is a demonstrator package used to compare baseline retrieval-augmented generation against a grounded approach built on top of the wider document-resolution system.

Its role is not to be the primary product surface, but to provide a controlled way of exploring how deterministic document resolution affects downstream retrieval and answer generation. In practice, this package is where the repository moves from extraction and resolution into demonstrable RAG behaviour.

## Purpose

The purpose of this package is to make architectural differences visible.

Rather than treating the document-resolution engine as an isolated NLP component, this package applies it in a downstream retrieval setting so that behaviour can be compared more concretely. The main comparison is between:

- a baseline RAG path that operates over document chunks in a more conventional way,
- a grounded path that carries forward deterministic document-resolution output before retrieval and answer generation.

This makes it easier to test and demonstrate questions such as:

- whether meaning is preserved more reliably,
- whether definition drift becomes visible,
- whether downstream behaviour becomes easier to audit,
- whether the system can answer, warn, retry, or abstain more deliberately.

## What lives in this package

At a high level, this package contains:

- baseline and grounded RAG pipelines,
- chunking and retrieval components,
- answer-generation components,
- embedding integration,
- pipeline composition helpers,
- bounded agentic review/orchestration support,
- settings and shared common models.

## Package structure

```text
src/rag_demo/
├── agentic/        # Bounded review/orchestration over grounded evidence
├── answering/      # Demo answer generation components
├── chunking/       # Chunking implementations
├── common/         # Shared models and types
├── composition/    # Pipeline assembly and embedder wiring
├── contracts/      # Interfaces and package contracts
├── embeddings/     # Embedding integrations
├── pipelines/      # Baseline and grounded pipeline implementations
├── retrieval/      # Retrieval backends and helpers
├── scenarios/      # Demo scenarios and supporting setup
├── settings.py     # Package settings
└── __init__.py
````

## Key areas

### `pipelines/`

This is the clearest starting point for understanding the package.

It contains the baseline and grounded pipeline implementations that the demonstrator is built around. These pipelines represent the main comparison the package exists to surface.

### `agentic/`

Contains the bounded reviewer/orchestrator logic used on the grounded path.

This is not intended to be a fully open-ended agent framework. Its role is narrower: inspect grounded evidence, determine whether it appears sufficient or ambiguous, and decide whether the system should proceed, retry once, warn, or abstain.

### `retrieval/`

Contains the retrieval implementations and supporting logic used by the demonstrator.

### `answering/`

Contains the answer-generation components used to turn retrieved evidence into responses in the demo flow.

### `composition/`

Contains package-level assembly logic, including embedder construction and pipeline setup.

### `chunking/`, `embeddings/`, and `common/`

These provide lower-level support for chunking, embedding integration, and shared package types and models.

## How this package relates to the rest of the repository

This package sits downstream of the main document-resolution engine.

It depends primarily on:

* `packages/document_resolution/` for the underlying resolution behaviour used in grounded flows,
* the wider repository configuration and service layers where demo integration is needed.

This package should be read as an application of the core system rather than as the core system itself. Its value is in showing how deterministic resolution affects retrieval behaviour, not in replacing the main resolution package.

## Design notes

This package is intentionally demonstrative in nature.

Part of its value lies not just in raw functionality, but in how clearly it exposes the difference between competing approaches. The package is meant to make behaviour visible: baseline drift, grounded stability, retry decisions, warning paths, and abstention should all be understandable rather than hidden behind a single undifferentiated answer path.

One of the more deliberate choices here is that the bounded review/orchestration logic is kept narrow. The reviewer is not supposed to independently resolve meanings or act as a free-form agent. Its role is to inspect structured grounded evidence and make a constrained decision about whether the evidence appears sufficient for the next step. That boundary matters to the package design.

As with other parts of the repository, there are areas that could be refined further. This is a demonstrator package rather than a polished standalone platform, and some assembly boundaries are lighter than they would be in a more mature dedicated RAG system. The current shape is intended to be honest to the repository’s goals: clear enough to demonstrate the architectural idea, without pretending to be a finished product in its own right.

## Design principles

This package was built around a few core ideas:

* make architectural differences visible rather than abstract,
* preserve deterministic document meaning where possible,
* keep grounded evidence inspectable,
* keep bounded reviewer behaviour constrained and explicit,
* prefer demonstration value and auditability over unnecessary complexity.

## Running and testing

Run tests from this package directory with:

```bash
poetry run pytest
```

Depending on how the demonstrator is being exercised, parts of the package may also be invoked through higher-level service or application flows elsewhere in the repository.

