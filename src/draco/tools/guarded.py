"""Guarded tool execution on top of the existing `app.agent.registry.CapabilityRegistry`.

The registry already validates input, applies permissions/timeouts/size caps and returns a
normalized `ToolResult`. This runtime adds what a harness needs around it: per-execution policy,
budget accounting, audit events, concurrency limits, bounded retries for non-mutating tools, and
untrusted-output flagging. Enforcement happens here, outside the model.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from typing import Any

from app.agent.contracts import ErrorCode, ToolError, ToolMeta, ToolResult, new_id
from app.agent.registry import Capability, CapabilityContext, CapabilityRegistry, TraceRecord
from app.agent.web import looks_injected

from draco.core.context import ExecutionContext
from draco.core.errors import ExecutionTimeout, PolicyViolation
from draco.harness.instrument import emit

_DENY_CODES = {"approval": ErrorCode.approval_required, "denied": ErrorCode.denied}


class ToolRuntime:
    def __init__(self, registry: CapabilityRegistry, *, services: Any = None,
                 evidence_factory: Callable[[ExecutionContext], Any] | None = None,
                 max_concurrency: int = 8, max_attempts: int = 2, retry_delay: float = 0.05):
        self.registry = registry
        self.services = services  # the app `Services` container handed to capability handlers
        self.evidence_factory = evidence_factory
        self._sem = asyncio.Semaphore(max_concurrency)
        self._locks: dict[str, asyncio.Lock] = {}
        self.max_attempts, self.retry_delay = max_attempts, retry_delay

    def specs(self, ctx: ExecutionContext, *, groups: set[str] | None = None) -> list[dict[str, Any]]:
        """Tool specs this execution may see. No tool is exposed without an explicit allow-list."""
        allowed = ctx.policy.allowed_tools
        if allowed is None:
            return []
        out = []
        for cap in self.registry.all():
            if (cap.name not in allowed or cap.name in ctx.policy.denied_tools or not cap.llm_exposed
                    or (cap.requires_approval and not ctx.policy.allow_approval_required)):
                continue
            if groups is not None and cap.group not in groups:
                continue
            if not cap.permissions <= ctx.policy.permissions:
                continue
            out.append(cap.spec())
        return out

    @staticmethod
    def _failure(name: str, ctx: ExecutionContext, code: ErrorCode, message: str, group: str = "unknown") -> ToolResult:
        meta = ToolMeta(trace_id=ctx.execution_id, execution_id=new_id(), capability_group=group, version="-")
        return ToolResult(tool_name=name, status="error", metadata=meta, error=ToolError(code=code, message=message))

    async def call(self, ctx: ExecutionContext, name: str, raw_args: Any, *, purpose: str = "") -> ToolResult:
        ctx.check()
        cap = self.registry.get(name)
        if cap is None:
            await emit(ctx, "tool.denied", tool=name, reason="unknown_tool")
            return self._failure(name, ctx, ErrorCode.not_found, f"Unknown capability '{name}'.")
        try:
            ctx.policy.authorize_tool(name, cap.permissions, requires_approval=cap.requires_approval)
            await emit(ctx, "tool.before", tool=name, purpose=purpose, group=cap.group,
                       mutates_state=cap.mutates_state)  # critical handlers may veto
        except PolicyViolation as exc:
            code = ErrorCode.approval_required if "approval" in exc.message else ErrorCode.denied
            await emit(ctx, "tool.denied", tool=name, reason=exc.message)
            return self._failure(name, ctx, code, exc.message, cap.group)
        ctx.budget.charge_tool_call()  # BudgetExhausted propagates: the run ends, not just the call
        cap_ctx = CapabilityContext(
            user_id=ctx.user_id or "anonymous", trace_id=ctx.execution_id, services=self.services,
            evidence=self.evidence_factory(ctx) if self.evidence_factory else None,
            permissions=ctx.policy.permissions, scratch={"draco_ctx": ctx})
        sink: list[TraceRecord] = []
        t0 = time.perf_counter()
        retries = 0
        lock = None if cap.parallel_safe else self._locks.setdefault(name, asyncio.Lock())
        while True:
            async with self._sem:
                if lock is not None:
                    async with lock:
                        res = await self._execute(ctx, cap, raw_args, cap_ctx, purpose, sink, retries)
                else:
                    res = await self._execute(ctx, cap, raw_args, cap_ctx, purpose, sink, retries)
            retryable = res.error is not None and res.error.retryable
            # mutating tools are never auto-retried: the first attempt may have taken effect
            if not retryable or cap.mutates_state or retries + 1 >= self.max_attempts:
                break
            retries += 1
            ctx.budget.charge_retry()
            await emit(ctx, "retry", tool=name, attempt=retries, error_code=res.error.code.value if res.error else None)
            await asyncio.sleep(self.retry_delay * 2 ** (retries - 1))
        injected = False
        if cap.untrusted_output and res.data is not None:
            injected = looks_injected(json.dumps(res.data, default=str))
            res.metadata.extra["untrusted"] = True
            if injected:
                res.metadata.extra["injection_suspected"] = True
        await emit(ctx, "tool.after", tool=name, status=res.status, group=cap.group, retries=retries,
                   duration_ms=round((time.perf_counter() - t0) * 1e3, 3), output_chars=res.metadata.output_chars,
                   error_code=res.error.code.value if res.error else None, untrusted=cap.untrusted_output,
                   injection_suspected=injected)
        return res

    async def _execute(self, ctx: ExecutionContext, cap: Capability, raw_args: Any, cap_ctx: CapabilityContext,
                       purpose: str, sink: list[TraceRecord], retry_count: int) -> ToolResult:
        left = ctx.time_left()
        coro = self.registry.execute(cap.name, raw_args, cap_ctx, purpose=purpose, sink=sink, retry_count=retry_count)
        if left is None:
            return await coro
        try:
            return await asyncio.wait_for(coro, timeout=max(left, 0.001))
        except TimeoutError:
            raise ExecutionTimeout("execution deadline exceeded during tool call") from None
