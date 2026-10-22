"""Guarded tool execution: validation, per-execution policy, authorization, budget accounting, audit
events, concurrency limits, bounded retries for non-mutating tools, and untrusted-output flagging.
Enforcement happens here, outside the model."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from pydantic import ValidationError

from draco.core.context import ExecutionContext
from draco.core.errors import ExecutionTimeout, PolicyViolation
from draco.harness.instrument import emit
from draco.security.audit import authorize
from draco.security.authorization import Resource
from draco.tools.contracts import (
    Empty,
    ErrorCode,
    Outcome,
    Tool,
    ToolErrorInfo,
    ToolFailure,
    ToolMeta,
    ToolResult,
    new_id,
)
from draco.tools.registry import ToolRegistry
from draco.tools.safety import looks_injected

logger = logging.getLogger("draco.tools")


def _summary_args(args: Any) -> dict[str, Any]:  # sizes only: arguments may be sensitive
    return {"keys": sorted(args)[:20]} if isinstance(args, dict) else {}


async def execute_tool(tool: Tool | None, name: str, raw_args: Any, ctx: ExecutionContext) -> ToolResult:
    """Validate and run one tool call, mapping every outcome to a ToolResult. Never leaks internals."""
    t0 = time.perf_counter()
    eid = new_id()

    def finish(status: str, data: Any = None, error: ToolErrorInfo | None = None, valid: bool = True,
               extra: dict[str, Any] | None = None) -> ToolResult:
        ms = round((time.perf_counter() - t0) * 1e3, 3)
        chars = len(json.dumps(data, default=str)) if data is not None else 0
        meta = ToolMeta(trace_id=ctx.execution_id, execution_id=eid, capability_group=tool.group if tool else "unknown",
                        version=tool.version if tool else "-", duration_ms=ms, output_chars=chars,
                        input_valid=valid, extra=extra or {})
        return ToolResult(tool_name=name, status=status, data=data, metadata=meta, error=error)  # type: ignore[arg-type]

    if tool is None:
        return finish("error", error=ToolErrorInfo(code=ErrorCode.not_found, message=f"Unknown tool '{name}'."), valid=False)
    try:
        if isinstance(raw_args, str):
            raw_args = json.loads(raw_args or "{}")
        args = tool.input_model.model_validate(raw_args or {})
    except (ValidationError, json.JSONDecodeError, TypeError) as exc:
        msg = ("Invalid arguments: " + "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
               if isinstance(exc, ValidationError) else "Arguments are not valid JSON.")
        return finish("error", error=ToolErrorInfo(code=ErrorCode.invalid_input, message=msg[:600]), valid=False)
    status, extra = "success", None
    try:
        out = await asyncio.wait_for(tool.handler(args, ctx), min(tool.timeout_s, 600))
        if isinstance(out, Outcome):
            status, extra, out = out.status, out.extra, out.data
        data = tool.output_model.model_validate(out).model_dump(mode="json") if tool.output_model else \
            (out.model_dump(mode="json") if hasattr(out, "model_dump") else out)
        if not isinstance(data, dict):  # handler contract: a JSON object
            logger.error("tool_output_invalid", extra={"tool": name})
            return finish("error", error=ToolErrorInfo(code=ErrorCode.internal, message="Tool returned an invalid result."))
    except Empty as e:
        return finish("empty", data=e.data)
    except TimeoutError:
        return finish("timeout", error=ToolErrorInfo(code=ErrorCode.timeout, retryable=True, message=f"{name} timed out."))
    except ToolFailure as e:
        return finish("error", error=ToolErrorInfo(code=e.code, message=e.message, retryable=e.retryable))
    except PolicyViolation as e:  # the tool's own authorization check refused: a denied call, not a crashed run
        code = ErrorCode.approval_required if "requires approval" in e.message else ErrorCode.denied
        return finish("error", error=ToolErrorInfo(code=code, message=e.message))
    except (asyncio.CancelledError, ExecutionTimeout):
        raise  # cancellation and deadline outcomes belong to the harness
    except ValidationError:
        logger.error("tool_output_invalid", extra={"tool": name})
        return finish("error", error=ToolErrorInfo(code=ErrorCode.internal, message="Tool returned an invalid result."))
    except PermissionError:
        return finish("error", error=ToolErrorInfo(code=ErrorCode.denied, message="Not permitted."))
    except Exception as exc:  # unexpected: never leak internals
        logger.error("tool_failed", extra={"tool": name, "error": type(exc).__name__, "execution_id": ctx.execution_id})
        return finish("error", error=ToolErrorInfo(code=ErrorCode.internal, message=f"{name} failed unexpectedly."))
    size = len(json.dumps(data, default=str))
    if size > tool.max_output_chars:
        return finish("error", error=ToolErrorInfo(code=ErrorCode.output_too_large,
                                                   message=f"Result too large ({size} chars); narrow the request."))
    return finish(status, data=data, extra=extra)


class ToolRuntime:
    def __init__(self, registry: ToolRegistry, *, max_concurrency: int = 8, max_attempts: int = 2,
                 retry_delay: float = 0.05):
        self.registry = registry
        self._sem = asyncio.Semaphore(max_concurrency)
        self._locks: dict[str, asyncio.Lock] = {}
        self.max_attempts, self.retry_delay = max_attempts, retry_delay

    def specs(self, ctx: ExecutionContext, *, groups: set[str] | None = None) -> list[dict[str, Any]]:
        """Tool specs this execution may see. No tool is exposed without an explicit allow-list, and
        approval-required tools are hidden unless the policy allows approval."""
        allowed = ctx.policy.allowed_tools
        if allowed is None:
            return []
        return [t.spec() for t in self.registry.all()
                if t.name in allowed and t.name not in ctx.policy.denied_tools and t.llm_exposed
                and (groups is None or t.group in groups) and t.permissions <= ctx.policy.permissions
                and (ctx.policy.allow_approval_required or not t.requires_approval)]

    @staticmethod
    def _failure(name: str, ctx: ExecutionContext, code: ErrorCode, message: str, group: str = "unknown") -> ToolResult:
        meta = ToolMeta(trace_id=ctx.execution_id, execution_id=new_id(), capability_group=group, version="-")
        return ToolResult(tool_name=name, status="error", metadata=meta, error=ToolErrorInfo(code=code, message=message))

    async def call(self, ctx: ExecutionContext, name: str, raw_args: Any, *, purpose: str = "") -> ToolResult:
        """Authorization (policy, then the harness authorizer) happens before any handler code runs, and
        before the call is charged to the budget."""
        ctx.check()
        tool = self.registry.get(name)
        if tool is None:
            await emit(ctx, "tool.denied", tool=name, reason="unknown_tool")
            return self._failure(name, ctx, ErrorCode.not_found, f"Unknown tool '{name}'.")
        try:
            ctx.policy.authorize_tool(name, tool.permissions, requires_approval=tool.requires_approval)
            await authorize(ctx, "tool.call", Resource("tool", name, attributes={
                "group": tool.group, "mutates_state": tool.mutates_state}))
            await emit(ctx, "tool.before", tool=name, purpose=purpose, group=tool.group,
                       mutates_state=tool.mutates_state)  # critical handlers may veto
        except PolicyViolation as exc:
            code = ErrorCode.approval_required if "approval" in exc.message else ErrorCode.denied
            await emit(ctx, "tool.denied", tool=name, reason=exc.message)
            return self._failure(name, ctx, code, exc.message, tool.group)
        ctx.budget.charge_tool_call()  # BudgetExhausted propagates: the run ends, not just the call
        t0 = time.perf_counter()
        retries = 0
        lock = None if tool.parallel_safe else self._locks.setdefault(name, asyncio.Lock())
        while True:
            async with self._sem:
                if lock is not None:
                    async with lock:
                        res = await self._attempt(ctx, tool, raw_args)
                else:
                    res = await self._attempt(ctx, tool, raw_args)
            retryable = res.error is not None and res.error.retryable
            # mutating tools are never auto-retried: the first attempt may have taken effect
            if not retryable or tool.mutates_state or retries + 1 >= self.max_attempts:
                break
            retries += 1
            ctx.budget.charge_retry()
            await emit(ctx, "retry", tool=name, attempt=retries, error_code=res.error.code.value if res.error else None)
            await asyncio.sleep(self.retry_delay * 2 ** (retries - 1))
        injected = False
        if tool.untrusted_output and res.data is not None:
            injected = looks_injected(json.dumps(res.data, default=str))
            res.metadata.extra["untrusted"] = True
            if injected:
                res.metadata.extra["injection_suspected"] = True
        await emit(ctx, "tool.after", tool=name, status=res.status, group=tool.group, retries=retries,
                   duration_ms=round((time.perf_counter() - t0) * 1e3, 3), output_chars=res.metadata.output_chars,
                   error_code=res.error.code.value if res.error else None, untrusted=tool.untrusted_output,
                   injection_suspected=injected, args=_summary_args(raw_args))
        return res

    async def _attempt(self, ctx: ExecutionContext, tool: Tool, raw_args: Any) -> ToolResult:
        left = ctx.time_left()
        coro = execute_tool(tool, tool.name, raw_args, ctx)
        if left is None:
            return await coro
        try:
            return await asyncio.wait_for(coro, timeout=max(left, 0.001))
        except TimeoutError:
            raise ExecutionTimeout("execution deadline exceeded during tool call") from None
