"""Run the existing LangGraph-based `app.agent.graph.AgentRunner` through the harness."""
from __future__ import annotations

from typing import Any

from draco.adapters.errors import map_app_error
from draco.core.context import ExecutionContext
from draco.core.errors import ContractError, ProviderError
from draco.core.protocols import Workload


def agent_runner_workload(runner: Any) -> Workload:
    """input: message string or an `AgentChatRequest` (dict ok). Returns the response as a dict.

    The runner builds its own LangGraph per request, so nothing is shared between executions. An
    `AgentChatResponse` with status "failed" (the runner's own canned failure) is surfaced as a
    ProviderError instead of a successful execution."""
    async def run(ctx: ExecutionContext, inp: Any) -> dict[str, Any]:
        from app.agent.schemas import AgentChatRequest
        if isinstance(inp, str):
            inp = {"message": inp}
        try:
            req = inp if isinstance(inp, AgentChatRequest) else AgentChatRequest.model_validate(inp)
        except Exception as exc:
            raise ContractError("invalid agent request") from exc
        ctx.check()
        try:
            resp = await runner.run(req, ctx.user_id or "anonymous")
        except Exception as exc:
            mapped = map_app_error(exc)
            if mapped is not None:
                raise mapped from exc
            raise
        out: dict[str, Any] = resp.model_dump(mode="json")
        if resp.status == "failed":
            raise ProviderError("agent run failed", retryable=False, details={"warnings": resp.warnings[:5]})
        if resp.status == "awaiting_approval":
            from draco.core.results import Waiting
            return Waiting("approval", out)  # type: ignore[return-value]
        return out
    return run
