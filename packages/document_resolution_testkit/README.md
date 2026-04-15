# document_resolution_testkit

## Overview

`document_resolution_testkit` is a shared testing support package for the wider repository.

Its purpose is to provide reusable fixtures, helpers, and test support code that can be used across packages and services rather than duplicated in each test tree. In its current form, it supports local and integration-style testing,
but its fuller intended role is as the foundation for more scenario-driven and behaviour-oriented test flows.

## Purpose

The main purpose of this package is to support shared testing infrastructure across the repository.

That includes:

- common fixtures,
- database test setup,
- shared helpers for test data and DB access,
- support code for behaviour-style testing.

More importantly, this package was designed with Gherkin-style testing in mind. The broader intention was not just to centralise a few fixtures, but to provide a proper base for feature-driven test scenarios using shared step implementations and reusable test context. It is already useful in its current state, but it is not yet being used to its full intended extent.

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

This area reflects the wider intended direction of the package.

The point of the test kit was not only to hold shared pytest fixtures, but to support reusable step implementations for behaviour-driven testing so that feature scenarios could be written more expressively and exercised consistently across the system.

## How this package relates to the rest of the repository

This package supports the rest of the repository rather than standing alone.

It is intended to reduce duplicated testing setup across:

* `packages/`
* `services/`

and to provide a shared base for both lower-level automated tests and higher-level scenario-driven testing.

In practice, it is currently used more as a fixture and helper package than as a full behaviour-test framework, but that is only part of the original intention.

## Design notes

This package is more about testing direction than architectural complexity.

In its current form, it is useful and actively used, particularly for shared fixtures and database-related testing support. However, it is not yet being used to its full intended potential. The larger goal was to make behaviour-driven testing easier to apply across the repository by giving Gherkin scenarios a reusable home for environment setup, shared context, and step implementations.

That matters because the value of a package like this increases as more tests move from isolated technical checks toward reusable behavioural scenarios. In other words, this package becomes more compelling when it is used not just to share fixtures, but to support feature-style testing with consistent steps and clearer system-level test flows.

So while the package is already useful, it should be read partly as groundwork for a richer testing approach rather than only as a small collection of helpers.

## Design principles

This package was built around a few core ideas:

* reduce duplicated test setup across packages and services,
* keep shared fixtures and helpers in one reusable place,
* support both technical and behaviour-oriented testing,
* make feature-style testing easier to grow over time,
* provide a base for Gherkin scenarios and reusable step implementations.

## Running and usage

This package is typically used indirectly by other test suites in the repository rather than run as a standalone application component.

Depending on the test flow, it may provide:

* shared pytest fixtures,
* database test setup,
* support for integration-style tests,
* support for behaviour-style tests.

## Current status

This is a small but intentionally reusable supporting package within the wider repository.

It is already useful for shared fixtures and test support, but its fuller value is really in enabling richer behaviour-driven testing over time. The original idea was not just to centralise plumbing, but to give Gherkin-based tests and shared step implementations a proper home as the repository’s testing approach matured.
