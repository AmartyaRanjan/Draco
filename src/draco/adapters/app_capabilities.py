"""Expose the existing `app.agent.registry.CapabilityRegistry` (21 capabilities) as Draco tools.

The app's executor is reused as-is: Draco validates, authorizes and budgets the call, then the app
registry runs its own validation, timeout, size cap and error mapping, and the outcome is translated
back to Draco's `ToolResult` contract."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from draco.core.context import ExecutionContext
from draco.core.errors import PolicyViolation
from draco.tools.contracts import Empty, ErrorCode, Outcome, Tool, ToolFailure
from draco.tools.registry import ToolRegistry


def register_app_capabilities(tools: ToolRegistry, app_registry: Any, *, services: Any,
                              evidence_factory: Callable[[ExecutionContext], Any] | None = None,
                              groups: set[str] | None = None) -> list[str]:
    """Register every (or every `groups`) app capability in `tools`. `services` is the app `Services`
    container handed to capability handlers. Returns the registered names."""
    from app.agent.registry import CapabilityContext

    names: list[str] = []
    for cap in app_registry.all():
        if groups is not None and cap.group not in groups:
            continue

        def make(cap: Any) -> Any:
            async def handler(args: BaseModel, ctx: ExecutionContext) -> Outcome:
                if not ctx.user_id:  # the app scopes data by user: no identity, no call
                    raise PolicyViolation(f"tool '{cap.name}' requires an authenticated user")
                cap_ctx = CapabilityContext(
                    user_id=ctx.user_id, trace_id=ctx.execution_id, services=services,
                    evidence=evidence_factory(ctx) if evidence_factory else None,
                    permissions=ctx.policy.permissions, scratch={"draco_ctx": ctx})
                res = await app_registry.execute(cap.name, args.model_dump(mode="json"), cap_ctx)
                if res.status == "timeout":
                    raise ToolFailure(ErrorCode.timeout, res.error.message if res.error else "timed out", retryable=True)
                if res.status == "error":
                    e = res.error
                    raise ToolFailure(ErrorCode(e.code.value) if e else ErrorCode.internal,
                                      e.message if e else "tool failed", bool(e and e.retryable))
                if res.status == "empty":
                    raise Empty(res.data)
                return Outcome(res.data or {}, res.status, res.metadata.extra)
            return handler

        tools.register(Tool(
            name=cap.name, description=cap.description, input_model=cap.input_model, handler=make(cap),
            group=cap.group, version=cap.version, permissions=frozenset(cap.permissions),
            mutates_state=cap.mutates_state, requires_approval=cap.requires_approval, parallel_safe=cap.parallel_safe,
            untrusted_output=cap.untrusted_output, llm_exposed=cap.llm_exposed,
            timeout_s=cap.timeout_s + 1.0, max_output_chars=cap.max_output_chars))
        names.append(cap.name)
    return names
