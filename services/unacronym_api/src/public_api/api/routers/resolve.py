import asyncio
import inspect
import math
import re
import time
from collections.abc import Iterable
from typing import Any

from fastapi import APIRouter, Response, status
from observability.config import REQ_ID_HEADER
from starlette.responses import JSONResponse

from public_api.api.response_types import build_responses
from public_api.api.types import APIDefinition, DBManagerDep, DefinitionCandidateLike, ResolverDep, SemaphoreDep
from public_api.core.settings import app_settings
from public_api.schemas.error import ErrorBody, ErrorCode, ErrorResponse
from public_api.schemas.resolve import ResolveOptions, ResolveRequest, ResolveResponse

router = APIRouter(prefix="/v1", tags=["Resolve"])

# Document response headers in OpenAPI
response_headers = {
    REQ_ID_HEADER: {
        "description": "Echoed or generated correlation id.",
        "schema": {"type": "string"},
    },
    "X-Input-Bytes": {
        "description": "Parsed request body size in bytes.",
        "schema": {"type": "integer"},
    },
    "X-Body-Limit-Bytes": {
        "description": "Current configured body size limit.",
        "schema": {"type": "integer"},
    },
    # Placeholders for future rate limiting (Epic 5)
    "X-RateLimit-Limit": {"description": "Requests allowed in window.", "schema": {"type": "integer"}},
    "X-RateLimit-Remaining": {"description": "Remaining in current window.", "schema": {"type": "integer"}},
    "X-RateLimit-Reset": {"description": "Epoch seconds until window resets.", "schema": {"type": "integer"}},
}

ACRO_PAREN_PATTERN = re.compile(r"\(([A-Z][A-Z0-9]{1,9})\)")  # simple, deterministic

def _bad_request(message: str, details: dict[str, Any] | None = None, http_status: int = 400) -> JSONResponse:
    return JSONResponse(
        status_code=http_status,
        content=ErrorResponse(
            error=ErrorBody(code=ErrorCode.BAD_REQUEST, message=message, details=details or {})
        ).model_dump()
    )

def _too_large(limit: int, actual: int) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        content=ErrorResponse(
            error=ErrorBody(
                code=ErrorCode.PAYLOAD_TOO_LARGE,
                message="Body/text too large.",
                details={"limit": limit, "actual": actual},
            )
        ).model_dump()
    )

def _svc_unavailable(reason: str) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(
            error=ErrorBody(
                code=ErrorCode.SERVICE_UNAVAILABLE,
                message="Service unavailable.",
                details={"reason": reason},
            )
        ).model_dump()
    )

# keep your helper but compute once, module-level (no per-request cost)
def _extract_max_len(model: type[ResolveRequest], field: str) -> Any:
    info = model.model_fields[field]
    # pydantic v2: constraints live in metadata (list of constraint objs)
    for meta in getattr(info, "metadata", []):
        if hasattr(meta, "max_length"):
            return meta.max_length
    # pydantic v1 fallback (harmless if v2)
    return getattr(info, "max_length", None)

TEXT_MAX_LEN = _extract_max_len(ResolveRequest, "text")


@router.post(
    "/resolve",
    response_model=ResolveResponse,
    responses=build_responses(success_status=200),
    summary="Resolve acronyms in raw text",
    description=(
        "Detect acronyms, propose definitions, and (optionally) enrich from a curated glossary. "
        "Offsets use Python-slice semantics (end exclusive). Supported locales: en-GB, en-US. "
        "Idempotent: does not mutate server state. Content-Encoding: gzip supported."
    ),
)
async def resolve_acronyms( # noqa: C901
    payload: ResolveRequest,
    response: Response,
    resolver: ResolverDep,
    semaphore: SemaphoreDep,
    dbm: DBManagerDep,
) -> JSONResponse | dict[str, Any]:
    definitions: list[APIDefinition] = []
    started = time.perf_counter()

    # Headers (size numbers reflect parsed model; body limit reflects config/middleware)
    text_bytes = len(payload.model_dump_json().encode("utf-8"))
    response.headers["X-Input-Bytes"] = str(text_bytes)
    response.headers["X-Body-Limit-Bytes"] = str(app_settings.MAX_BODY_BYTES)

    # Semantic validation: whitespace-only → 422
    if not payload.text.strip():
        err = ErrorResponse(
            error=ErrorBody(
                code=ErrorCode.UNPROCESSABLE_ENTITY,
                message="Text must not be empty.",
                details={"hint": "Provide non-empty 'text'"},
            )
        )
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=err.model_dump())

    if TEXT_MAX_LEN and len(payload.text) > TEXT_MAX_LEN:
        return _too_large(limit=TEXT_MAX_LEN, actual=len(payload.text))

    # Options: coerce/validate (Pydantic already validated bounds/types)
    opts = payload.options or ResolveOptions.model_validate({})
    if any(map(math.isnan, [float(opts.min_confidence)])):
        return _bad_request("Invalid numeric option.", {"field": "min_confidence"})

    # Concurrency cap
    if semaphore is not None and semaphore.locked() and semaphore._value == 0:  # noqa: SLF001 (internal field read)
        return _svc_unavailable("OVERLOADED")

    async def _call_resolver(acronym: str) -> Iterable[DefinitionCandidateLike]:
        from core.domain import Acronym
        res = resolver.resolve(Acronym(text=acronym), top_k=opts.max_definitions_per_acronym)
        if inspect.isawaitable(res):
            res = await res
        return res

    # Detect acronyms by simple paren rule; order by first occurrence
    matches = list(ACRO_PAREN_PATTERN.finditer(payload.text))
    # Build one acronym block per unique acronym ordered by first hit
    seen: set[str] = set()
    blocks: list[dict[str, Any]] = []
    for m in matches:
        ac = m.group(1)
        if ac in seen:
            continue
        seen.add(ac)

        first_occ = {"start": m.start(1), "end": m.end(1)}
        coro = _call_resolver(ac)

        if opts.include_glossary_enrichment:
            # assumes you whitelisted these tables when constructing DBManager
            table = "glossary_entries"
            row = dbm.select_one_dict(
                table_fqn=table,
                columns=["acronym", "definition", "source"],
                criteria=[("acronym", "", ac.lower())],  # see note below
            )
            if row:
                definitions.append({
                    "text": row["definition"],
                    "start": max(0, first_occ["start"] - opts.window_chars),
                    "end": first_occ["end"],
                    "confidence": 1.0,
                    "source": row.get("source") or "glossary",
                })

        try:
            # 2) If a global semaphore exists, use it to throttle; otherwise just run.
            if semaphore is not None:
                async with semaphore:
                    results = await asyncio.wait_for(
                        coro,
                        timeout=app_settings.REQUEST_TIMEOUT_MS / 1000.0,
                    )
            else:
                results = await asyncio.wait_for(
                    coro,
                    timeout=app_settings.REQUEST_TIMEOUT_MS / 1000.0,
                )

        except asyncio.TimeoutError:
            err = ErrorResponse(
                error=ErrorBody(
                    code=ErrorCode.SERVICE_UNAVAILABLE,
                    message="Resolution timed out.",
                    details={"timeout_ms": app_settings.REQUEST_TIMEOUT_MS, "acronym": ac},
                )
            )
            # Decide policy: either bail out (return 503) or record a per-acronym error and continue.
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=err.model_dump(),
            )

        except Exception as exc:
            # Map unexpected resolver failures (same policy decision as above).
            err = ErrorResponse(
                error=ErrorBody(
                    code=ErrorCode.SERVICE_UNAVAILABLE,
                    message="Resolution failed.",
                    details={"acronym": ac, "reason": str(exc)},
                )
            )
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=err.model_dump(),
            )

        # Map core DefinitionCandidates -> API definitions
        definitions = [
            {
                "text": c.text,
                # window hint only; adjust later with real extraction
                "start": max(0, first_occ["start"] - opts.window_chars),
                "end": first_occ["end"],  # placeholder end; core will supply in future stories
                "confidence": float(c.score),
                "source": "extracted",
            }
            for c in results
            if c.score >= float(opts.min_confidence)
        ]

        # Occurrences list (optional): all matches of this acronym
        occs = [{"start": mm.start(1), "end": mm.end(1)} for mm in matches if mm.group(1) == ac]

        block: dict[str, Any] = {
            "acronym": ac,
            "first_occurrence": first_occ,
            "definitions": definitions,
        }
        if opts.return_occurrences:
            block["occurrences"] = occs
        # TODO Glossary enrichment stub stays empty until 2.5
        blocks.append(block)

    # Meta
    processing_ms = int((time.perf_counter() - started) * 1000)
    model_version = "plainera-core@dev"

    return {
        "acronyms": blocks,
        "meta": {
            "processing_ms": processing_ms,
            "model_version": model_version,
            "input_chars": len(payload.text),
        },
    }
