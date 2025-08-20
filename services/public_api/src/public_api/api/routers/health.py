from fastapi import APIRouter, Request

router = APIRouter()

@router.get("/healthz", summary="Liveness")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}

@router.get("/readyz", summary="Readiness")
async def readyz() -> dict[str, str]:
    return {"status": "ready"}


@router.post("/_echo")
async def echo(request: Request) -> dict[str, int]:
    data = await request.body()  # force body read
    return {"len": len(data)}
