# document_resolution_testkit

## Overview

`document_resolution_testkit` is a shared testing support package for the wider repository.

Its purpose is to provide reusable fixtures, helpers, and test support code that can be used across packages and services rather than duplicated in each test tree. In its current form, it supports shared fixture setup, database-related test helpers, and integration-style testing support.

It was also designed with a broader testing direction in mind: providing a base for more scenario-driven and behaviour-oriented test flows over time.

## Purpose

The main purpose of this package is to support shared testing infrastructure across the repository.

That includes:

- common fixtures,
- database test setup,
- shared helpers for test data and DB access,
- support code for behaviour-style testing.

A more deliberate part of the design was to make Gherkin-style testing easier to support in one place. The aim was not only to centralise a few fixtures, but to provide a reusable home for feature-driven test scenarios, shared step implementations, and common test context where that style of testing is useful.

## What lives in this package

At a high level, this package contains:

- shared test fixtures,
- test configuration,
- database-related test helpers,
- common test utilities,
- behaviour-test environment setup,
- step implementation support for Gherkin-style tests.

## Package structure

```text
test_kit/
├── behave_env.py           # Behaviour-test environment setup
├── common.py               # Shared test helpers
├── config.py               # Test configuration
├── fixtures.py             # Shared pytest fixtures
├── helpers/
│   ├── data.py             # Test data helpers
│   └── db.py               # Database test helpers
├── step_implementations/
│   └── db_impl.py          # Example/shared behaviour test steps
└── __init__.py
````

## Key areas

### `fixtures.py`

This is the most directly used part of the package in its current form.

It provides shared fixtures that can be bridged into local test trees so that packages and services can use common setup without duplicating the same plumbing in multiple places.

### `helpers/`

Contains supporting helpers for test data and database-related test operations.

### `behave_env.py`

Contains behaviour-test environment support intended for Gherkin-style testing flows.

### `step_implementations/`

This reflects the wider intended direction of the package.

The test kit was not designed only to hold shared pytest fixtures, but also to support reusable step implementations for behaviour-driven testing so that feature scenarios could be written more expressively and exercised more consistently across the system.

## How this package relates to the rest of the repository

This package supports the rest of the repository rather than standing alone.

It is intended to reduce duplicated testing setup across:

* `packages/`
* `services/`

and to provide a shared base for both lower-level automated tests and higher-level scenario-driven testing.

In practice, it is currently used more as a fixture and helper package than as a full behaviour-test framework, but that is only part of the original design intent.

## Design notes

This package is less about complex architecture and more about making the repository’s testing support reusable and easier to grow.

In its current form, it is already useful for shared fixtures and database-related test support. At the same time, part of its value is in the testing direction it enables: giving behaviour-driven tests a reusable home for environment setup, shared context, and step implementations rather than scattering that setup across multiple test trees.

That matters because the package becomes more valuable as tests move beyond isolated technical checks and toward reusable behavioural scenarios. In other words, its strongest long-term role is not only to centralise plumbing, but to support clearer system-level and feature-level test flows.

## Design principles

This package was built around a few core ideas:

* reduce duplicated test setup across packages and services,
* keep shared fixtures and helpers in one reusable place,
* support both technical and behaviour-oriented testing,
* make feature-style testing easier to grow over time,
* provide a base for Gherkin scenarios and reusable step implementations.

## Running and usage

This package is typically used indirectly by other test suites in the repository rather than as a standalone application component.

Depending on the test flow, it may provide:

* shared pytest fixtures,
* database test setup,
* support for integration-style tests,
* support for behaviour-style tests.
