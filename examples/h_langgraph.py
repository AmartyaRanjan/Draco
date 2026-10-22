"""Example H: an external framework (LangGraph) integrated through an adapter. Needs `pip install draco[agents]`.
LangGraph owns graph execution; Draco adds identity, budgets, deadlines, events and interrupt/resume."""
import asyncio

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from draco import Harness
from draco.adapters.langgraph import graph_workload


class State(TypedDict, total=False):
    request: str
    approved: str


def ask_human(state: State) -> State:
    return {"approved": interrupt({"approve": state["request"]})}


builder = StateGraph(State)
builder.add_node("ask_human", ask_human)
builder.add_edge(START, "ask_human")
builder.add_edge("ask_human", END)
graph = builder.compile(checkpointer=InMemorySaver())


async def main():
    h = Harness()
    h.register("approval_flow", graph_workload(graph, to_state=lambda x: x if isinstance(x, Command) else {"request": x}))
    first = await h.run("approval_flow", "refund order 17")
    print("status:", first.status.value, "| waiting on:", first.output)
    done = await h.resume(first.execution_id, input=Command(resume="yes"))
    print("status:", done.status.value, "| result:", done.output, "| same execution id:", done.execution_id == first.execution_id)

asyncio.run(main())
