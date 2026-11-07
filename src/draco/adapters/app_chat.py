"""Run the starter app's `ChatOrchestrator` (`/chat`, rag and agent modes) through the harness.

The execution id becomes the orchestrator's request id (`metadata.request_id` of the response and the
retrieval trace prefix). Per-request objects come from `ctx.metadata` when supplied:

* `metadata["orchestrator"]`: the orchestrator instance to use; otherwise the one bound at registration.
* `metadata["emit"]`: an async callable receiving `/chat/stream` events (start, token, tool_call,
  tool_result, done, error). The orchestrator emits `done`/`error` itself.
"""
from __future__ import annotations

from typing import Any

from draco.adapters.app_agent import _mapped
from draco.core.context import ExecutionContext
from draco.core.errors import ConfigurationError, ContractError, PolicyViolation
from draco.core.protocols import Workload


def chat_workload(orchestrator: Any = None) -> Workload:
    """input: message string or a `ChatRequest` (dict ok). Returns the `ChatResponse` as a dict."""
    async def run(ctx: ExecutionContext, inp: Any) -> dict[str, Any]:
        from app.inference.schemas import ChatRequest
        if isinstance(inp, str):
            inp = {"message": inp}
        try:
            req = inp if isinstance(inp, ChatRequest) else ChatRequest.model_validate(inp)
        except Exception as exc:
            raise ContractError("invalid chat request") from exc
        ctx.check()
        if not ctx.user_id:  # conversations are owned per user: never default to a shared identity
            raise PolicyViolation("chat execution requires an authenticated user")
        impl = ctx.metadata.get("orchestrator") or orchestrator
        if impl is None:
            raise ConfigurationError("no chat orchestrator configured for this workload")
        try:
            resp = await impl.run(req, ctx.user_id, ctx.metadata.get("emit"), request_id=ctx.execution_id)
        except Exception as exc:
            mapped = _mapped(exc)
            if mapped is exc:
                raise
            raise mapped from exc
        out: dict[str, Any] = resp.model_dump(mode="json")
        return out
    return run
