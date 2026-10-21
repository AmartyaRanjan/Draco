"""Helpers workloads use so model calls, retries and steps are budgeted, evented and traceable."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from draco.core.context import ExecutionContext
from draco.core.errors import DracoError
from draco.core.events import Event, EventBus

T = TypeVar("T")


class ReconciliationRequired(DracoError):
    """A non-idempotent step started but never recorded completion; its side effect is unknown."""
    code = "reconciliation_required"


def _bus(ctx: ExecutionContext) -> EventBus | None:
    s = ctx.services
    return s.get("events") if s is not None and s.has("events") else None


async def emit(ctx: ExecutionContext, name: str, **data: Any) -> None:
    bus = _bus(ctx)
    if bus is not None:
        await bus.emit(Event(name, ctx.execution_id, ctx.parent_execution_id, ctx.correlation_id, data))


async def call_model(ctx: ExecutionContext, fn: Callable[..., Awaitable[T]], *args: Any, provider: str = "",
                     model: str = "", usage_of: Callable[[T], int | None] | None = None, **kwargs: Any) -> T:
    """Budget-charged, cancellable, evented model call. `model.before` handlers may be critical (veto)."""
    ctx.check()
    ctx.budget.charge_model_call()
    await emit(ctx, "model.before", provider=provider, model=model)
    t0 = time.perf_counter()
    try:
        res = await fn(*args, **kwargs)
    except Exception as exc:
        await emit(ctx, "model.failed", provider=provider, model=model, error=type(exc).__name__,
                   latency_ms=round((time.perf_counter() - t0) * 1e3, 3))
        raise
    tokens = usage_of(res) if usage_of else None
    if tokens:
        ctx.budget.add_tokens(tokens)
    await emit(ctx, "model.after", provider=provider, model=model, tokens=tokens,
               latency_ms=round((time.perf_counter() - t0) * 1e3, 3))
    return res


def _retryable(exc: BaseException) -> bool:
    return isinstance(exc, DracoError) and exc.retryable


async def call_with_retry(ctx: ExecutionContext, fn: Callable[[], Awaitable[T]], *, max_attempts: int = 3,
                          base_delay: float = 0.05,
                          retry_on: Callable[[BaseException], bool] = _retryable) -> T:
    """Retry only failures the predicate calls retryable; each retry is charged to the budget."""
    attempt = 0
    while True:
        ctx.check()
        try:
            return await fn()
        except Exception as exc:
            attempt += 1
            if attempt >= max_attempts or not retry_on(exc):
                raise
            ctx.budget.charge_retry()
            await emit(ctx, "retry", attempt=attempt, error=type(exc).__name__)
            await asyncio.sleep(base_delay * 2 ** (attempt - 1))


async def step(ctx: ExecutionContext, name: str, fn: Callable[[], Awaitable[T]], *, idempotent: bool = False) -> T:
    """Run `fn` once per execution. On resume a completed step returns its recorded result instead of
    re-running; a step that started but never completed is re-run only if `idempotent`, otherwise
    ReconciliationRequired is raised, because a side effect may or may not have happened.
    Results must be JSON-serializable."""
    store = ctx.services.get("store")
    rec = await store.get_step(ctx.execution_id, name)
    if rec is not None and rec.status == "done":
        await emit(ctx, "step.replayed", step=name)
        return rec.data
    if rec is not None and rec.status == "started" and not idempotent:
        await emit(ctx, "step.reconciliation_required", step=name)
        raise ReconciliationRequired(f"step '{name}' started before an interruption and is not marked "
                                     f"idempotent; verify its effect externally before resuming",
                                     details={"step": name})
    await store.begin_step(ctx.execution_id, name)
    await emit(ctx, "step.started", step=name)
    out = await fn()
    await store.complete_step(ctx.execution_id, name, out)
    await emit(ctx, "step.completed", step=name)
    return out
