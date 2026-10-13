"""Liveness, readiness and metrics. Liveness never touches dependencies (a restart would not fix them);
readiness does, and gates traffic. Probes are cached briefly so a probe storm cannot hammer the stores."""
import asyncio
import inspect
import hmac
import time

import httpx
from fastapi import APIRouter, Header, HTTPException, Response
from fastapi.responses import JSONResponse

from app.core import observability as obs
from app.core.http_settings import get_platform_settings

router = APIRouter(tags=["ops"])
_cache: tuple[float, dict] | None = None
PROBE_TIMEOUT_S = 2.0


def _redis():
    from app.core.queue import redis_conn
    redis_conn.ping()


def _qdrant():
    from app.db.qdrant import COLLECTION_NAME, get_qdrant_client
    if not get_qdrant_client().collection_exists(COLLECTION_NAME):
        raise RuntimeError("collection missing")


def _opensearch():
    from app.db.opensearch import INDEX_NAME, get_opensearch_client
    if not get_opensearch_client().indices.exists(index=INDEX_NAME):
        raise RuntimeError("index missing")


async def _llm():
    from app.inference.config import get_inference_settings
    s = get_inference_settings()
    async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_S) as c:
        r = await c.get(f"{s.base_url.rstrip('/')}/models", headers={"Authorization": f"Bearer {s.api_key}"})
        r.raise_for_status()


async def _probe(name, fn):
    t0 = time.perf_counter()
    try:
        if inspect.iscoroutinefunction(fn):
            await asyncio.wait_for(fn(), PROBE_TIMEOUT_S)
        else:
            await asyncio.wait_for(asyncio.to_thread(fn), PROBE_TIMEOUT_S)
        ok, err = True, None
    except Exception as exc:
        ok, err = False, type(exc).__name__
    obs.READY.labels(name).set(1 if ok else 0)
    return name, {"ok": ok, "ms": round((time.perf_counter() - t0) * 1000), **({"error": err} if err else {})}


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/health/ready")
async def ready():
    global _cache
    ttl = get_platform_settings().ready_cache_s
    if _cache and time.monotonic() - _cache[0] < ttl:
        body = _cache[1]
    else:
        checks = dict(await asyncio.gather(*(_probe(n, f) for n, f in
                      (("redis", _redis), ("qdrant", _qdrant), ("opensearch", _opensearch), ("llm", _llm)))))
        body = {"status": "ready" if all(c["ok"] for c in checks.values()) else "unavailable", "checks": checks}
        _cache = (time.monotonic(), body)
    return JSONResponse(body, status_code=200 if body["status"] == "ready" else 503)


@router.get("/metrics", include_in_schema=False)
def metrics(authorization: str | None = Header(default=None)):
    token = get_platform_settings().metrics_token
    if token:
        supplied = (authorization or "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(supplied.encode(), token.encode()):
            raise HTTPException(401, "Invalid or missing metrics token")
    body, ctype = obs.render_metrics()
    return Response(body, media_type=ctype)
