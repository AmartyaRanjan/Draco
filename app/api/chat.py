import asyncio
import hmac
import json
import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.core import concurrency, ratelimit
from app.inference.config import get_inference_settings
from app.inference.generation import ChatOrchestrator
from app.inference.memory import SqliteConversationStore
from app.inference.schemas import ChatRequest, InferenceError
from app.retrieval.context import TokenCounter
from app.retrieval.service import get_service

logger = logging.getLogger("rag.inference")
router = APIRouter()
_orchestrator: ChatOrchestrator | None = None


def get_orchestrator() -> ChatOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        s = get_inference_settings()
        # Budget prompt tokens with the *generation* model's tokenizer (retrieval's default is BGE-M3's).
        _orchestrator = ChatOrchestrator(get_service(), s, SqliteConversationStore(s.store_path), counter=TokenCounter(s.model))
    return _orchestrator


async def current_user(request: Request, x_api_key: str | None = Header(default=None)) -> str:
    """Authenticate at the application boundary; returns the user id bound to the API key.
    Failed attempts are throttled per client address so keys cannot be brute-forced."""
    s = get_inference_settings()
    if s.auth_disabled:
        return "anonymous"
    ip = ratelimit.client_ip(request)
    blocked = await ratelimit.check("authfail", ip, cost=0)  # peek: a locked-out address is refused even with a valid key
    if not blocked.allowed:
        raise ratelimit.reject("authfail", blocked, "Too many failed authentication attempts")
    if x_api_key:
        for key, user in s.api_keys.items():
            if hmac.compare_digest(key.encode(), x_api_key.encode()):
                return user
    await ratelimit.check("authfail", ip)
    raise HTTPException(401, "Invalid or missing API key")


heavy_user = ratelimit.limited("heavy", current_user)  # LLM-backed routes
read_user = ratelimit.limited("read", current_user)
upload_user = ratelimit.limited("upload", current_user)


def _http(exc: InferenceError) -> HTTPException:
    return HTTPException(exc.status_code, f"{exc} ({exc.code})")


@router.post("/chat")
async def chat(req: ChatRequest, user: str = Depends(heavy_user),
               orch: ChatOrchestrator = Depends(get_orchestrator)):
    try:
        async with concurrency.llm_slot():
            return (await orch.run(req, user)).model_dump()
    except HTTPException:
        raise
    except InferenceError as exc:
        raise _http(exc) from exc
    except Exception as exc:
        logger.exception("chat_unexpected")
        raise HTTPException(500, "Internal error while answering") from exc


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request, user: str = Depends(heavy_user),
                      orch: ChatOrchestrator = Depends(get_orchestrator)):
    """SSE events: start | token | tool_call | tool_result | done | error. `done.response` is the
    validated final answer and supersedes the streamed text; incomplete runs are never persisted."""
    await concurrency.acquire_llm_slot()  # 503 now, before the stream starts, if the model is saturated
    queue: asyncio.Queue = asyncio.Queue()

    async def emit(ev):
        await queue.put(ev)

    async def work():
        try:
            await orch.run(req, user, emit)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass  # run() already emitted an error event
        finally:
            concurrency.release_llm_slot()
            await queue.put(None)

    task = asyncio.create_task(work())

    async def events():
        try:
            while True:
                try:
                    ev = await asyncio.wait_for(queue.get(), 1.0)
                except asyncio.TimeoutError:
                    if await request.is_disconnected():
                        break
                    continue
                if ev is None:
                    break
                yield f"event: {ev['event']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
        finally:
            if not task.done():
                task.cancel()  # client gone: cancels the upstream vLLM request, no persistence

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
