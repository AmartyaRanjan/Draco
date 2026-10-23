import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.agent import router as agent_router
from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.query import router as query_router
from app.api.routes import router
from app.core.http_settings import get_platform_settings
from app.core.middleware import PlatformMiddleware
from app.core.observability import configure_logging
from app.agent.config import get_agent_settings
from app.db.registry import init_db
from app.inference.config import get_inference_settings

logger = logging.getLogger("rag.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    p = get_platform_settings()
    configure_logging(p.log_level, p.log_json)
    # Fail at boot, not on the first request: bad limits, missing API keys, unsafe production flags.
    p.validate(auth_disabled=get_inference_settings().auth_disabled)
    get_agent_settings().validate(production=p.production)
    init_db()
    from app.core.queue import ingestion_queue
    from app.ingestion.dispatch import reconcile_ingestion_outbox

    async def reconcile():
        while True:
            try:
                await asyncio.to_thread(reconcile_ingestion_outbox, ingestion_queue)
            except Exception:
                logger.exception("ingestion_outbox_reconciliation_failed")
            await asyncio.sleep(30)

    reconciliation_task = asyncio.create_task(reconcile())
    logger.info("startup", extra={"env": p.env, "rate_limit": p.rate_limit_enabled, "cors_origins": list(p.cors_origins)})
    try:
        yield
    finally:
        reconciliation_task.cancel()
        await asyncio.gather(reconciliation_task, return_exceptions=True)
    logger.info("shutdown")


app = FastAPI(
    title="Draco RAG API",
    description="Document ingestion, hybrid retrieval, grounded chat and a tool-using agent.",
    version="1.0.0",
    lifespan=lifespan,
)

_cors = get_platform_settings().cors_origins
if _cors:
    app.add_middleware(CORSMiddleware, allow_origins=list(_cors), allow_methods=["GET", "POST"],
                       allow_headers=["x-api-key", "content-type", "x-request-id"],
                       expose_headers=["x-request-id", "retry-after", "x-ratelimit-limit", "x-ratelimit-remaining"])
app.add_middleware(PlatformMiddleware)  # added last = outermost: it sees every request, including CORS preflights

app.include_router(health_router)
app.include_router(router)
app.include_router(query_router)
app.include_router(chat_router)
app.include_router(agent_router)
