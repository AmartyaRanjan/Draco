"""Run any compiled LangGraph graph as a Draco workload (LangGraph stays the graph engine)."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from draco.core.context import ExecutionContext
from draco.core.protocols import Workload


def graph_workload(graph: Any, *, to_state: Callable[[Any], Any] = lambda x: x,
                   from_state: Callable[[Any], Any] = lambda s: s, recursion_limit: int = 25) -> Workload:
    """`graph` is a compiled graph with `ainvoke`. The execution id becomes the LangGraph thread id,
    so checkpointer state (if the graph has one) is keyed to the Draco execution; cancellation and
    deadlines are enforced by the harness around the `ainvoke` task."""
    async def run(ctx: ExecutionContext, inp: Any) -> Any:
        ctx.check()
        cfg = {"configurable": {"thread_id": ctx.execution_id}, "recursion_limit": recursion_limit,
               "metadata": {"draco_execution_id": ctx.execution_id, "draco_correlation_id": ctx.correlation_id}}
        return from_state(await graph.ainvoke(to_state(inp), cfg))
    return run
