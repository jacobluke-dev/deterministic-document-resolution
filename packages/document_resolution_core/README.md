````md
# document_resolution_core

## Overview

`document_resolution_core` is a small supporting package within the wider repository.

It contains lower-level shared components that sit beneath the main document-resolution engine, primarily around database manager construction, sink wiring, mapping utilities, and a small set of general-purpose helpers. It is intentionally much lighter than `packages/document_resolution` and should be read as supporting infrastructure rather than as a standalone domain package.

## Purpose

The purpose of this package is to hold shared lower-level components that would otherwise either be duplicated across the repository or sit awkwardly inside higher-level packages.

In practice, this package currently focuses on:

- database manager construction,
- SQLAlchemy session and connection helpers,
- sink construction and sink-related mapping,
- a small number of utility functions.

This package is deliberately narrow in scope.

## What lives in this package

At a high level, this package contains:

- database manager primitives and supporting factories,
- SQLAlchemy session helpers,
- sink construction and sink implementations,
- payload-to-row mapping support,
- shared type aliases for lower-level DB manager components,
- small general-purpose utilities.

## Package structure

```text
src/document_resolution_core/
├── db_manager/
│   ├── connection.py     # DBManager and core connection/session handling
│   ├── dbm_factory.py    # DB manager construction helpers
│   ├── mappers.py        # Payload-to-row mapping helpers
│   ├── sessions.py       # SQLAlchemy sessionmaker/session utilities
│   ├── sink_factory.py   # Sink registration and sink construction
│   ├── sinks.py          # Sink implementations
│   ├── types.py          # Shared lower-level types
│   └── README.md
├── utils/
│   └── utils.py          # Small shared utility helpers
└── __init__.py
````

## Key areas

### `db_manager/`

This is the main purpose of the package.

It contains the lower-level database-related support used elsewhere in the repository, including:

* DB manager construction,
* session setup,
* sink construction,
* mapper wiring,
* related shared types.

This package is intentionally concerned with support mechanics rather than higher-level document-resolution logic.

### `utils/`

A small collection of general-purpose helpers that did not justify living inside a more domain-specific package.

## How this package relates to the rest of the repository

This package is used as a supporting layer beneath higher-level parts of the repository, especially:

* `packages/document_resolution/`
* `services/document_resolution_api/`

It should not be read as the main entry point for understanding the system. For that, the more useful places to start are the top-level repository README, the API service, and the main `document_resolution` package.

## Design notes

This package is intentionally lightweight, and that is both a strength and a limitation.

On the positive side, it keeps lower-level support concerns separate from the main document-processing package. On the other hand, because its scope is narrow, some abstractions here are relatively thin. That is a known characteristic of the package rather than an unknown issue.

In earlier iterations, this area carried more structural weight. It has since been reduced to a smaller and more honest support package; I think that is an improvement.

There are still places where the boundaries could be refined further, but the package is now closer to its real role in the repository: shared support code rather than a major architectural centre of gravity.

## Design principles

This package was built around a few simple ideas:

* keep lower-level support concerns out of higher-level packages where possible,
* keep database and sink construction explicit,
* prefer reusable wiring over duplicated setup,
* keep supporting abstractions as small and honest as possible,
* avoid turning shared infrastructure code into a larger framework than the repository actually needs.


## Current status

This is a small supporting package within a larger public-facing repository.

It is not intended to be a large standalone library. Its value is in providing a cleaner home for shared lower-level mechanics that support the rest of the system.

A future rename may be appropriate, as the package is now more representative of database management and supporting infrastructure than the broader `document_resolution_core` name suggests.
