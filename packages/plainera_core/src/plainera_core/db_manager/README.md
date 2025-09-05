
# `db_manager`

Database sink utilities for structured logging.

## Purpose

This package provides sink implementations that persist structured log payloads (dicts) into different backends (e.g. Postgres via SQLAlchemy). Sinks are pluggable: you can attach one or more to the `emit`/`emit_async` logging pipeline.

## Core Concepts

### `SqlAlchemyModelSink`

* Writes each payload to a configured SQLAlchemy model/table.
* Designed for durability and correctness.
* Used as the default sink for storing logs in Postgres.

### `CompositeSink`

* **Fan-out**: forwards every payload to *all* configured sinks.
* Use when you want logs duplicated to multiple destinations.
* Example: write logs to Postgres **and** to stdout for local debugging.

```python
sink = CompositeSink([SqlAlchemyModelSink(...), ConsoleSink()])
sink.enqueue({"event": "user_signup", "level": "info"})
```

### `RouterSink`

* **Conditional routing**: forwards a payload only to sinks whose predicate matches.
* Use when you want to direct different types of logs to different places.
* Example: send errors to Postgres, but access logs only to a file.

```python
error_only = lambda p: p.get("level") == "error"
info_only = lambda p: p.get("level") == "info"

sink = RouterSink([
    (error_only, SqlAlchemyModelSink(...)),
    (info_only, ConsoleSink()),
])
sink.enqueue({"event": "startup", "level": "info"})  # goes only to ConsoleSink
```

## When to Use Them

* **Single sink (most cases):** If you only need to persist logs to one backend (e.g. Postgres), use `SqlAlchemyModelSink` directly. This keeps things simple.
* **CompositeSink:** If you want the *same* log in multiple places (DB, file, stdout, external shipper).
* **RouterSink:** If you want *different* logs in different places (errors to DB, requests to S3, etc.).

## Caveats

* Both `CompositeSink` and `RouterSink` currently use `asyncio.create_task` to schedule sink operations in a fire-and-forget manner. In short-lived scripts, this can lose logs because the process may exit before tasks complete.
* For production durability, prefer calling sinks via `emit_async(...)`, which awaits sink writes.

---
