"""Pieces shared by both answer paths: `/chat` (`ChatOrchestrator`) and `/agent/chat` (`AgentRunner`).

* model calls: `GovernedModel` + `invoke_model` (streaming or not, `<think>` filtering, timings)
* the retrieval tool call: `scoped_retrieve` (document-scope check, then the hybrid pipeline)
* tool governance: `begin_tool` / `end_tool` (policy, authorizer, budget, events)
* citations: `resolve_markers` / `strip_markers` (only labels that resolve to evidence survive)

Both paths run as Draco workloads (see `app.agent.harness`). The harness makes its `ExecutionContext`
current for the duration of the workload, so every model call, tool call and retrieval made through
these helpers is charged to the execution budget, checked against the execution's `Policy` and
authorizer, cancellable and evented, whichever loop made it. Outside a harness (unit tests, scripts)
the helpers are plain pass-throughs.
"""

from __future__ import annotations

import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from draco.core.context import ExecutionContext, current_context
from draco.core.errors import DracoError, PolicyViolation
from draco.harness.instrument import call_model
from draco.harness.instrument import emit as draco_emit
from draco.security.audit import authorize
from draco.security.authorization import Resource

Emit = Callable[[dict], Awaitable[None]]
THINK = re.compile(r"<think>.*?</think>", re.S)


def strip_think(text: Any) -> str:
    return THINK.sub("", text if isinstance(text, str) else "").strip()


class ThinkFilter:
    """Defensive: drop <think>…</think> from streamed text if the server parser is misconfigured."""

    def __init__(self):
        self.buf, self.inside = "", False

    def feed(self, text: str) -> str:
        self.buf += text
        out = ""
        while True:
            if self.inside:
                j = self.buf.find("</think>")
                if j < 0:
                    self.buf = self.buf[-8:]
                    return out
                self.buf, self.inside = self.buf[j + 8:], False
            else:
                j = self.buf.find("<think>")
                if j >= 0:
                    out += self.buf[:j]
                    self.buf, self.inside = self.buf[j + 7:], True
                    continue
                keep = 6 if "<think>"[:1] in self.buf[-6:] else 0  # hold a possible partial tag
                out += self.buf[:len(self.buf) - keep]
                self.buf = self.buf[len(self.buf) - keep:]
                return out


def is_framework_error(exc: BaseException) -> bool:
    """Budget, policy, cancellation and deadline outcomes belong to the harness: never remap them into
    application errors (InferenceError / tool results)."""
    return isinstance(exc, DracoError)


# ---- model calls ----------------------------------------------------------------------------------
def _usage_tokens(usage: dict | None) -> int | None:
    if not usage:
        return None
    total = usage.get("total_tokens") or (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
    return int(total) or None


class GovernedModel:
    """Wraps a LangChain chat model. Inside a Draco execution each `ainvoke`/`astream` is a budgeted,
    cancellable model call with `model.before|after|failed` events; outside one it is the plain client.
    Every other attribute is the wrapped client's."""

    def __init__(self, llm: Any, *, provider: str = "app.inference", model: str = ""):
        self._llm, self._provider, self._model = llm, provider, model

    @property
    def inner(self) -> Any:
        return self._llm

    def bind_tools(self, *args: Any, **kwargs: Any) -> GovernedModel:
        return GovernedModel(self._llm.bind_tools(*args, **kwargs), provider=self._provider, model=self._model)

    async def ainvoke(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        ctx = current_context()
        if ctx is None:
            return await self._llm.ainvoke(messages, *args, **kwargs)
        return await call_model(ctx, self._llm.ainvoke, messages, *args, provider=self._provider,
                                model=self._model,
                                usage_of=lambda m: _usage_tokens(getattr(m, "usage_metadata", None)), **kwargs)

    async def astream(self, messages: Any, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        ctx = current_context()
        if ctx is None:
            async for chunk in self._llm.astream(messages, *args, **kwargs):
                yield chunk
            return
        ctx.check()
        ctx.budget.charge_model_call()
        await draco_emit(ctx, "model.before", provider=self._provider, model=self._model, streaming=True)
        t0, tokens = time.perf_counter(), 0
        try:
            async for chunk in self._llm.astream(messages, *args, **kwargs):
                tokens += _usage_tokens(getattr(chunk, "usage_metadata", None)) or 0
                yield chunk
        except Exception as exc:
            await draco_emit(ctx, "model.failed", provider=self._provider, model=self._model,
                             error=type(exc).__name__, latency_ms=round((time.perf_counter() - t0) * 1e3, 3))
            raise
        if tokens:
            ctx.budget.add_tokens(tokens)
        await draco_emit(ctx, "model.after", provider=self._provider, model=self._model, tokens=tokens or None,
                         latency_ms=round((time.perf_counter() - t0) * 1e3, 3), streaming=True)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)


@dataclass
class ModelCall:
    message: Any  # the accumulated AIMessage(Chunk)
    text: str  # content with <think> removed
    tool_calls: list
    usage: dict
    finish_reason: str | None
    ttft_ms: float | None
    total_ms: float
    mean_inter_token_ms: float | None = None
    tokens_streamed: int = 0


async def invoke_model(llm: Any, messages: list, *, emit: Emit | None = None) -> ModelCall:
    """One model call. With `emit`, the response is streamed and each visible text delta is emitted as
    `{"event": "token", "text": ...}` (thinking blocks are filtered out). Errors propagate unchanged."""
    t0, first, last, acc, gaps, streamed = time.perf_counter(), None, None, None, [], 0
    if emit is None:
        acc = await llm.ainvoke(messages)
        first = time.perf_counter()
    else:
        flt = ThinkFilter()
        async for chunk in llm.astream(messages):
            acc = chunk if acc is None else acc + chunk
            text = chunk.content if isinstance(chunk.content, str) else ""
            text = flt.feed(text) if text else ""
            if text:
                now = time.perf_counter()
                if first is None:
                    first = now
                elif last is not None:
                    gaps.append(now - last)
                last = now
                streamed += 1
                await emit({"event": "token", "text": text})
    end = time.perf_counter()
    if acc is None:
        from app.inference.schemas import InferenceError

        raise InferenceError("Empty response from model", "llm_empty", 502)
    return ModelCall(
        message=acc,
        text=strip_think(acc.content),
        tool_calls=list(getattr(acc, "tool_calls", None) or []),
        usage=dict(getattr(acc, "usage_metadata", None) or {}),
        finish_reason=(getattr(acc, "response_metadata", None) or {}).get("finish_reason"),
        ttft_ms=round((first - t0) * 1e3, 3) if first else None,
        total_ms=round((end - t0) * 1e3, 3),
        mean_inter_token_ms=round(sum(gaps) / len(gaps) * 1e3, 3) if gaps else None,
        tokens_streamed=streamed,
    )


def thinking_exhausted(call: ModelCall, thinking: bool) -> bool:
    """Thinking consumed the whole output budget: both paths retry once without thinking."""
    return thinking and not call.text and call.finish_reason == "length"


# ---- retrieval ------------------------------------------------------------------------------------
class ScopeDenied(PermissionError):
    """A model-chosen document scope tried to widen the scope the caller allowed."""


def effective_scope(requested: list[str] | None, allowed: list[str] | None) -> list[str] | None:
    if requested and allowed and not set(requested) <= set(allowed):
        raise ScopeDenied("document outside the permitted set")
    return requested or allowed


async def scoped_retrieve(retrieval: Any, query: str, requested: list[str] | None, allowed: list[str] | None,
                          *, request_id: str, principal: str) -> Any:
    """The retrieval tool call both paths make: narrow (never widen) the document scope, run the hybrid
    pipeline as `principal`, and record a `retrieval.after` event on the current execution."""
    scope = effective_scope(requested, allowed)
    ctx = current_context()
    if ctx is not None:
        ctx.check()
    out = await retrieval.retrieve(query, scope, request_id, principal)
    if ctx is not None:
        sel = getattr(out, "selection", None)
        await draco_emit(ctx, "retrieval.after", selected=len(sel.selected) if sel and sel.selected else 0,
                         degraded=list(getattr(out, "degraded", []) or []), scoped=scope is not None)
    return out


def filenames_for(retrieval: Any, document_ids: Iterable[str], cache: dict[str, str | None]) -> dict[str, str | None]:
    for d in document_ids:
        if d not in cache:
            cache[d] = retrieval.filename_for(d)
    return cache


# ---- tool governance ------------------------------------------------------------------------------
@dataclass
class ToolGate:
    ctx: ExecutionContext | None
    name: str
    t0: float = field(default_factory=time.perf_counter)


async def begin_tool(name: str, *, permissions: Iterable[str] = (), requires_approval: bool = False,
                     group: str = "app", mutates_state: bool = False, purpose: str = "") -> ToolGate:
    """Authorize and charge one tool call against the current execution, in ToolRuntime's order: policy
    (allow-list, permissions, approval), the harness authorizer (`tool.call`), `tool.before` (critical
    handlers may veto), budget. Raises PolicyViolation (after a `tool.denied` event) when refused;
    BudgetExhausted propagates so the run ends."""
    ctx = current_context()
    if ctx is None:
        return ToolGate(None, name)
    ctx.check()
    try:
        ctx.policy.authorize_tool(name, frozenset(permissions), requires_approval=requires_approval)
        await authorize(ctx, "tool.call", Resource("tool", name, attributes={"group": group,
                                                                           "mutates_state": mutates_state}))
        await draco_emit(ctx, "tool.before", tool=name, purpose=purpose, group=group, mutates_state=mutates_state)
    except PolicyViolation as exc:
        await draco_emit(ctx, "tool.denied", tool=name, reason=exc.message)
        raise
    ctx.budget.charge_tool_call()
    return ToolGate(ctx, name)


async def end_tool(gate: ToolGate, status: str, *, error_code: str | None = None, output_chars: int = 0) -> None:
    if gate.ctx is not None:
        await draco_emit(gate.ctx, "tool.after", tool=gate.name, status=status, error_code=error_code,
                         output_chars=output_chars, duration_ms=round((time.perf_counter() - gate.t0) * 1e3, 3))


# ---- citations ------------------------------------------------------------------------------------
def strip_markers(answer: str, labels: Iterable[str]) -> str:
    """Never let an unverifiable reference reach the client."""
    for label in labels:
        answer = answer.replace(f"[{label}]", "")
    return answer


def resolve_markers(answer: str, pattern: re.Pattern, lookup: Callable[[str], Any]) -> tuple[str, list, list[str]]:
    """Resolve each distinct `[label]` through `lookup` (the run's evidence). Returns the answer with
    unresolved markers removed, the resolved items in first-mention order, and the unresolved labels."""
    valid, invalid, seen = [], [], set()
    for label in pattern.findall(answer):
        if label in seen:
            continue
        seen.add(label)
        hit = lookup(label)
        if hit:
            valid.append(hit)
        else:
            invalid.append(label)
    return strip_markers(answer, invalid), valid, invalid
