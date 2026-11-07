"""HTTP surface of the agent layer. Approval/reject/ingest are the human-in-the-loop step: they are
authenticated API calls that write the approval record; no model-callable capability can.

`/agent/chat`, `/agent/chat/stream` and `/agent/executions/{id}/resume` run as Draco workloads
(`app.agent.harness`), with the same identity, policy, budget, deadline, cancellation, idempotency and
tracing as `/chat`."""

import logging
from typing import Literal

from draco.core.results import Status
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from app.agent import artifacts as A
from app.agent.catalog import build_registry
from app.agent.evidence import EvidenceStore
from app.agent.graph import AgentRunner
from app.agent.harness import AGENT, AGENT_RESUME, agent_policy, run_workload
from app.agent.registry import CapabilityContext
from app.agent.schemas import AgentChatRequest, AgentChatResponse
from app.core import concurrency
from app.api.chat import IdempotencyKey, heavy_user, read_user, result_output, sse_response, stream_error
from app.inference.config import get_inference_settings
from app.inference.memory import SqliteConversationStore
from app.inference.schemas import InferenceError
from app.retrieval.service import get_service

logger = logging.getLogger("rag.agent")
router = APIRouter(prefix="/agent", tags=["agent"])
_runner: AgentRunner | None = None


def get_runner() -> AgentRunner:
    global _runner
    if _runner is None:
        from app.agent.wiring import build_services

        inf = get_inference_settings()
        svc, adapter = build_services(inf, get_service())
        _runner = AgentRunner(svc, build_registry(), adapter, inf, SqliteConversationStore(inf.store_path))
    return _runner


def _err(code) -> int:
    return {
        "invalid_input": 422,
        "not_found": 404,
        "denied": 403,
        "approval_required": 409,
        "not_configured": 503,
        "timeout": 504,
        "upstream_error": 502,
        "unsafe_url": 422,
        "limit_exceeded": 413,
        "conflict": 409,
    }.get(code, 500)


async def _cap(runner: AgentRunner, name: str, args: dict, user: str):
    ctx = CapabilityContext(
        user_id=user,
        trace_id="api",
        services=runner.svc,
        evidence=EvidenceStore(),
        permissions=frozenset({"read", "write", "generate"}),
    )
    res = await runner.reg.execute(name, args, ctx, purpose="api request")
    if not res.ok:
        raise HTTPException(_err(res.error.code.value), f"{res.error.message} ({res.error.code.value})")
    return res


async def _run_agent(req: AgentChatRequest, user: str, runner: AgentRunner, emit=None, key: str | None = None):
    return await run_workload(AGENT, req.model_dump(mode="json"), user,
                              policy=agent_policy(runner.reg, allow_web=req.allow_web), impl_key="runner",
                              impl=runner, emit=emit, idempotency_key=key, session_id=req.conversation_id)


@router.post("/chat")
async def agent_chat(
    req: AgentChatRequest,
    user: str = Depends(heavy_user),
    runner: AgentRunner = Depends(get_runner),
    idempotency_key: str | None = IdempotencyKey,
):
    try:
        async with concurrency.llm_slot():
            res = await _run_agent(req, user, runner, key=idempotency_key)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("agent_chat_unexpected")
        raise HTTPException(500, "Internal error while answering") from exc
    return result_output(res)


@router.post("/chat/stream")
async def agent_chat_stream(
    req: AgentChatRequest,
    request: Request,
    user: str = Depends(heavy_user),
    runner: AgentRunner = Depends(get_runner),
    idempotency_key: str | None = IdempotencyKey,
):
    """SSE events: start | step | tool_call | tool_result | token | revision, then exactly one terminal
    event: done (`response`), awaiting_approval (`execution_id`, `pending_approvals`, `response`; resume
    with POST /agent/executions/{id}/resume) or error. Tokens are streamed for the final answer of
    question-answering runs only; `revision` means the streamed text is being replaced, and the terminal
    `response` (validated, citations resolved) always supersedes streamed text."""
    await concurrency.acquire_llm_slot()  # 503 now, before the stream starts, if the model is saturated

    async def work(emit):
        res = await _run_agent(req, user, runner, emit, idempotency_key)
        body = None
        try:
            body = result_output(res)
        except HTTPException:
            await emit(stream_error(res))
            return
        if res.replayed:
            await emit({"event": "start", "request_id": res.execution_id, "replayed": True,
                        "conversation_id": (body.get("metadata") or {}).get("conversation_id")})
        if res.status == Status.waiting:
            await emit({"event": "awaiting_approval", "execution_id": res.execution_id,
                        "pending_approvals": body.get("pending_approvals", []), "response": body})
        else:
            await emit({"event": "done", "response": body})

    return sse_response(work, request)


@router.get("/capabilities")
async def capabilities(user: str = Depends(read_user), runner: AgentRunner = Depends(get_runner)):
    caps = runner.adapter.caps
    return {
        "capabilities": [c.describe() for c in runner.reg.all()],
        "groups": runner.reg.groups(),
        "unconfigured": runner.svc.missing(),
        "runtime": {
            "model": caps.model,
            "profile": caps.profile,
            "supports_thinking_toggle": caps.supports_toggle,
            "fixed_mode": caps.fixed_mode,
            "template_declares_flag": caps.template_declares_flag,
            "probe_verified": caps.probe_verified,
            "notes": caps.notes,
        },
    }


class ApproveBody(BaseModel):
    ingest: bool = Field(default=False, description="start ingestion immediately after approving")


@router.post("/sources/{source_id}/approve")
async def approve_source(
    source_id: str,
    body: ApproveBody | None = None,
    user: str = Depends(read_user),
    runner: AgentRunner = Depends(get_runner),
):
    try:
        ap = runner.svc.store.approve_source(
            source_id, user, approve=True, ttl_hours=runner.svc.settings.approval_ttl_hours
        )
    except KeyError:
        raise HTTPException(404, "Source not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    out = {"source_id": source_id, "decision": ap["decision"], "expires_at": ap["expires_at"]}
    if body and body.ingest:
        out["ingestion"] = (
            await _cap(runner, "ingest_approved_source", {"source_id": source_id}, user)
        ).model_dump()
    return out


@router.post("/sources/{source_id}/reject")
async def reject_source(
    source_id: str, user: str = Depends(read_user), runner: AgentRunner = Depends(get_runner)
):
    try:
        ap = runner.svc.store.approve_source(source_id, user, approve=False)
    except KeyError:
        raise HTTPException(404, "Source not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    return {"source_id": source_id, "decision": ap["decision"]}


@router.post("/sources/{source_id}/ingest")
async def ingest_source(
    source_id: str, user: str = Depends(read_user), runner: AgentRunner = Depends(get_runner)
):
    return (await _cap(runner, "ingest_approved_source", {"source_id": source_id}, user)).model_dump()


@router.get("/sources/{source_id}")
async def source_provenance(
    source_id: str, user: str = Depends(read_user), runner: AgentRunner = Depends(get_runner)
):
    return (await _cap(runner, "inspect_source_provenance", {"source_id": source_id}, user)).model_dump()


@router.get("/artifacts/{artifact_id}")
async def download_artifact(
    artifact_id: str, user: str = Depends(read_user), runner: AgentRunner = Depends(get_runner)
):
    try:
        rec, path = runner.svc.artifact_store.resolve(artifact_id, user)  # owner, expiry, file present
    except A.ArtifactError:
        raise HTTPException(404, "Artifact not found")
    return FileResponse(
        path,
        media_type=A.MIME.get(rec["fmt"], "application/octet-stream"),
        filename=rec["filename"],
        headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "private, no-store"},
    )


class FeedbackBody(BaseModel):
    target_type: str = Field(pattern="^(answer|artifact|source)$")
    target_id: str = Field(min_length=1, max_length=256)
    rating: int = Field(ge=1, le=5)
    comment: str | None = Field(default=None, max_length=2000)


@router.post("/feedback")
async def feedback(
    body: FeedbackBody, user: str = Depends(read_user), runner: AgentRunner = Depends(get_runner)
):
    return (await _cap(runner, "record_feedback", body.model_dump(), user)).model_dump()


class ResumeBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "reject"]


class PendingApprovalResponse(BaseModel):
    source_id: str
    title: str | None = None
    url: str
    trust: str | None = None
    approve: str | None = None


class AgentExecutionResponse(BaseModel):
    execution_id: str
    status: Literal["running", "waiting", "completed", "failed", "expired"]
    waiting_for_approval: bool
    pending_approvals: list[PendingApprovalResponse]
    resumable: bool
    created_at: float
    updated_at: float
    expires_at: float
    result: AgentChatResponse | None = None
    error: str | None = None


@router.get("/executions/{execution_id}")
async def get_execution(
    execution_id: str,
    user: str = Depends(read_user),
    runner: AgentRunner = Depends(get_runner),
) -> AgentExecutionResponse:
    """Return one execution only when it belongs to the authenticated principal."""
    try:
        record = await runner.get_execution(execution_id, user, tenant_id=user)
    except InferenceError as exc:
        raise HTTPException(exc.status_code, f"{exc} ({exc.code})") from exc
    response = record["response"] or {}
    status = "running" if record["status"] == "resuming" else record["status"]
    return AgentExecutionResponse(
        execution_id=record["execution_id"],
        status=status,
        waiting_for_approval=status == "waiting",
        pending_approvals=response.get("pending_approvals", []),
        resumable=status == "waiting",
        created_at=record["created_at"],
        updated_at=record["updated_at"],
        expires_at=record["expires_at"],
        result=response if status in ("completed", "failed") else None,
        error=record["error"],
    )


@router.post("/executions/{execution_id}/resume")
async def resume_execution(
    execution_id: str,
    body: ResumeBody,
    user: str = Depends(heavy_user),
    runner: AgentRunner = Depends(get_runner),
) -> AgentChatResponse:
    """Submit a validated approval decision and resume the interrupted execution (Draco workload
    `app.agent.resume`; ownership is checked against the authenticated principal before the graph runs)."""
    try:
        res = await run_workload(AGENT_RESUME, {"execution_id": execution_id, "decision": body.decision}, user,
                                 policy=agent_policy(runner.reg), impl_key="runner", impl=runner)
    except Exception as exc:
        logger.exception("agent_resume_unexpected")
        raise HTTPException(500, "Internal error while resuming execution") from exc
    return result_output(res)
