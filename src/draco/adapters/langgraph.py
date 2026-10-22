"""Run any compiled LangGraph graph as a Draco workload. LangGraph stays the graph engine; install with
`pip install draco[agents]` (this module itself imports nothing from langgraph).

Semantics:
- thread id: the Draco execution id, so a checkpointer's state is keyed to the execution. `Harness.resume`
  keeps the execution id, so a resumed run continues the same LangGraph thread.
- identity: tenant, user, session and correlation ids are passed as run metadata (for tracing only; they
  are never read back from graph state).
- recursion: `recursion_limit` bounds graph supersteps; exceeding it is reported as `BudgetExhausted`.
- interrupts: a result containing LangGraph's `__interrupt__` is returned as `Waiting`; continue with
  `harness.resume(id, input=Command(resume=...))`.
- cancellation/deadline: enforced by the harness around the `ainvoke` task (cooperative at awaits).
- budgets: model calls made *inside* the graph are only counted if the graph's nodes call
  `draco.harness.call_model`; the adapter cannot see them otherwise."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from draco.core.context import ExecutionContext
from draco.core.errors import BudgetExhausted
from draco.core.protocols import Workload
from draco.core.results import Waiting


def graph_workload(graph: Any, *, to_state: Callable[[Any], Any] = lambda x: x,
                   from_state: Callable[[Any], Any] = lambda s: s, recursion_limit: int = 25) -> Workload:
    """`graph` is a compiled graph with `ainvoke`."""
    async def run(ctx: ExecutionContext, inp: Any) -> Any:
        ctx.check()
        cfg = {"configurable": {"thread_id": ctx.execution_id}, "recursion_limit": recursion_limit,
               "metadata": {"draco_execution_id": ctx.execution_id, "draco_correlation_id": ctx.correlation_id,
                            "draco_tenant_id": ctx.tenant_id, "draco_session_id": ctx.session_id}}
        try:
            out = await graph.ainvoke(to_state(inp), cfg)
        except Exception as exc:
            if type(exc).__name__ == "GraphRecursionError":
                raise BudgetExhausted("graph_steps", recursion_limit) from exc
            raise
        if isinstance(out, dict) and out.get("__interrupt__"):
            return Waiting("graph interrupted", [getattr(i, "value", i) for i in out["__interrupt__"]])
        return from_state(out)
    return run
