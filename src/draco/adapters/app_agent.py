"""Run the existing LangGraph-based `app.agent.graph.AgentRunner` through the harness.

The execution id becomes the runner's request id, so the Draco execution, the LangGraph thread, the
app's execution record (`GET /agent/executions/{id}`) and the response's `metadata.request_id` are one
identity. Per-request objects come from `ctx.metadata` when the caller supplies them:

* `metadata["runner"]`: the runner instance to use (the HTTP layer passes its dependency-injected one);
  otherwise the runner bound at registration.
* `metadata["emit"]`: an async callable receiving stream events (`start`, `step`, `tool_call`, ...).
"""
from __future__ import annotations

import logging
from typing import Any

from draco.adapters.errors import map_app_error
from draco.core.context import ExecutionContext
from draco.core.errors import ConfigurationError, ContractError, DracoError, PolicyViolation, ProviderError
from draco.core.protocols import Workload
from draco.core.results import Waiting

logger = logging.getLogger("draco.adapters.app")


def _runner(ctx: ExecutionContext, bound: Any) -> Any:
    runner = ctx.metadata.get("runner") or bound
    if runner is None:
        raise ConfigurationError("no agent runner configured for this workload")
    return runner


def _require_user(ctx: ExecutionContext) -> str:
    if not ctx.user_id:  # the runner scopes sessions and evidence by user: never default to a shared identity
        raise PolicyViolation("agent execution requires an authenticated user")
    return ctx.user_id


def _mapped(exc: Exception) -> Exception:
    """App exceptions -> typed Draco errors; framework errors pass through; anything else is logged."""
    if isinstance(exc, DracoError):
        return exc
    mapped = map_app_error(exc)
    if mapped is None:
        logger.exception("app_workload_unexpected", extra={"error": type(exc).__name__})
        return exc
    return mapped


def _outcome(resp: Any) -> Any:
    """completed -> dict; awaiting_approval -> Waiting(payload); the runner's canned failure -> ProviderError
    whose `details["response"]` keeps the client-facing response body."""
    out: dict[str, Any] = resp.model_dump(mode="json")
    if resp.status == "failed":
        raise ProviderError("agent run failed", retryable=False,
                            details={"warnings": resp.warnings[:5], "response": out})
    if resp.status == "awaiting_approval":
        return Waiting("approval", out)
    return out


def agent_runner_workload(runner: Any = None) -> Workload:
    """input: message string or an `AgentChatRequest` (dict ok). Returns the response as a dict, or
    `Waiting("approval", response)` when the run stopped at the human-approval interrupt.

    An `AgentChatResponse` with status "failed" (the runner's own canned failure) is surfaced as a
    ProviderError instead of a successful execution."""
    async def run(ctx: ExecutionContext, inp: Any) -> Any:
        from app.agent.schemas import AgentChatRequest
        if isinstance(inp, str):
            inp = {"message": inp}
        try:
            req = inp if isinstance(inp, AgentChatRequest) else AgentChatRequest.model_validate(inp)
        except Exception as exc:
            raise ContractError("invalid agent request") from exc
        ctx.check()
        user = _require_user(ctx)
        impl = _runner(ctx, runner)
        try:
            resp = await impl.run(req, user, tenant_id=ctx.tenant_id or user, request_id=ctx.execution_id,
                                  emit=ctx.metadata.get("emit"))
        except Exception as exc:
            mapped = _mapped(exc)
            if mapped is exc:
                raise
            raise mapped from exc
        return _outcome(resp)
    return run


def agent_resume_workload(runner: Any = None) -> Workload:
    """input: `{"execution_id": str, "decision": "approve" | "reject"}`. Resumes a waiting agent execution
    from its durable LangGraph checkpoint. Ownership (user and tenant of this execution), decision
    validation, resume claims and expiry are enforced by the runner before the graph is invoked."""
    async def run(ctx: ExecutionContext, inp: Any) -> Any:
        if not isinstance(inp, dict) or not isinstance(inp.get("execution_id"), str):
            raise ContractError("input must be {'execution_id': str, 'decision': 'approve' | 'reject'}")
        ctx.check()
        user = _require_user(ctx)
        impl = _runner(ctx, runner)
        try:
            resp = await impl.resume(inp["execution_id"], inp.get("decision"), user, tenant_id=ctx.tenant_id or user)
        except Exception as exc:
            mapped = _mapped(exc)
            if mapped is exc:
                raise
            raise mapped from exc
        return _outcome(resp)
    return run
