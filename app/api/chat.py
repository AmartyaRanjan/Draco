import asyncio
import hmac
import json
import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from draco.core.results import ExecutionResult, Status

from app.agent.harness import CHAT, chat_policy, run_workload
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


IdempotencyKey = Header(default=None, alias="Idempotency-Key", min_length=1, max_length=200,
                        description="replays the recorded response of a completed (or approval-waiting) run")


def result_output(res: ExecutionResult):
    """The response body of a Draco execution, or the HTTPException the old handlers raised for the same
    failure: app errors keep their status code and `message (code)` text; framework outcomes map to
    504 (deadline), 429 (budget / tenant concurrency), 403 (policy), 422 (contract), 503 (cancelled)."""
    if res.status in (Status.completed, Status.waiting):
        return res.output
    e = res.error
    details = e.details if e else {}
    if res.status == Status.failed and isinstance(details.get("response"), dict):
        return details["response"]  # the agent's own failed response is a 200 body, as before
    if e is not None and isinstance(details.get("status_code"), int):
        raise HTTPException(details["status_code"], f"{e.message} ({details.get('stage') or e.code})")
    if res.status == Status.timeout:
        raise HTTPException(504, "The request exceeded its execution deadline (timeout)")
    if res.status == Status.budget_exhausted:
        raise HTTPException(429, f"{e.message if e else 'budget exhausted'} (budget_exhausted)",
                            headers={"Retry-After": "5"})
    if res.status == Status.cancelled:
        raise HTTPException(503, "The request was cancelled (cancelled)")
    if e is not None and e.code == "policy":
        raise HTTPException(403, "Not permitted by the execution policy (policy)")
    if e is not None and e.code == "contract":
        raise HTTPException(422, f"{e.message} (contract)")
    logger.error("execution_failed", extra={"execution_id": res.execution_id, "error": e.type if e else None})
    raise HTTPException(500, "Internal error while answering")


def stream_error(res: ExecutionResult) -> dict:
    """The SSE `error` event for a failed execution (same status and text as the JSON endpoint)."""
    try:
        result_output(res)
    except HTTPException as exc:
        detail = str(exc.detail)
        code = detail[detail.rfind("(") + 1:-1] if detail.endswith(")") else "internal"
        return {"event": "error", "code": code, "message": detail, "status_code": exc.status_code}
    return {"event": "error", "code": "internal", "message": "Internal error", "status_code": 500}


def sse_response(work, request: Request) -> StreamingResponse:
    """Run `work(emit)` in a task and stream what it emits as SSE. A client disconnect cancels the task
    (and with it the Draco execution and the upstream model request); the LLM slot is released either way."""
    queue: asyncio.Queue = asyncio.Queue()

    async def emit(ev):
        await queue.put(ev)

    async def runner():
        try:
            await work(emit)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("stream_unexpected")
            await queue.put({"event": "error", "code": "internal", "message": "Internal error"})
        finally:
            concurrency.release_llm_slot()
            await queue.put(None)

    task = asyncio.create_task(runner())

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
                task.cancel()  # client gone: cancels the execution and the upstream vLLM request, no persistence

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def _run_chat(req: ChatRequest, user: str, orch: ChatOrchestrator, emit=None, key: str | None = None):
    return await run_workload(CHAT, req.model_dump(mode="json"), user, policy=chat_policy(),
                              impl_key="orchestrator", impl=orch, emit=emit, idempotency_key=key,
                              session_id=req.conversation_id)


@router.post("/chat")
async def chat(req: ChatRequest, user: str = Depends(heavy_user),
               orch: ChatOrchestrator = Depends(get_orchestrator), idempotency_key: str | None = IdempotencyKey):
    """Runs as the Draco workload `app.chat` (identity, policy, budget, deadline, idempotency, tracing)."""
    try:
        async with concurrency.llm_slot():
            res = await _run_chat(req, user, orch, key=idempotency_key)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("chat_unexpected")
        raise HTTPException(500, "Internal error while answering") from exc
    return result_output(res)


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request, user: str = Depends(heavy_user),
                      orch: ChatOrchestrator = Depends(get_orchestrator),
                      idempotency_key: str | None = IdempotencyKey):
    """SSE events: start | token | tool_call | tool_result | done | error. `done.response` is the
    validated final answer and supersedes the streamed text; incomplete runs are never persisted."""
    await concurrency.acquire_llm_slot()  # 503 now, before the stream starts, if the model is saturated

    async def work(emit):
        terminal = False

        async def tracked(ev):
            nonlocal terminal
            terminal = terminal or ev.get("event") in ("done", "error")
            await emit(ev)

        res = await _run_chat(req, user, orch, tracked, idempotency_key)
        if terminal:
            return  # the orchestrator emitted done/error itself
        if res.status == Status.completed:  # an idempotent replay: the run did not execute again
            meta = (res.output or {}).get("metadata", {})
            await emit({"event": "start", "request_id": meta.get("request_id"),
                        "conversation_id": meta.get("conversation_id"), "replayed": True})
            await emit({"event": "done", "response": res.output})
        else:  # refused or failed before the orchestrator ran (policy, budget, deadline)
            await emit(stream_error(res))

    return sse_response(work, request)

