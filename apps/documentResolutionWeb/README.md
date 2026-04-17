# documentResolutionWeb

## Overview

`documentResolutionWeb` is the Next.js web application for the Document Resolution repository.

It provides a browser-based interface for interacting with the API and demonstrating the resolution workflows exposed by the wider system.

## Current scope

The current UI is primarily a demonstrator rather than a full product interface.

At present, the main surfaced workflow is the acronym-resolution demo page, which allows local testing of API-backed resolution behaviour from the browser.

## Local usage

In the normal repository workflow, this app is run via Docker Compose from the monorepo root.

The local demo page is available at:

`http://127.0.0.1:3001/demo`

## Relationship to the repository

This application should usually be read alongside:

- `services/document_resolution_api/` for the backing FastAPI service,
- `packages/document_resolution/` for the core resolution logic,
- the top-level `README.md` for overall repository setup and usage.

## Notes

This app is still relatively lightweight compared with the rest of the repository and is intended mainly as a thin UI layer over the underlying API and resolution engine.
