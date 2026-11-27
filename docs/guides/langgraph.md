# Wrap a LangGraph graph

If you already build agents as [LangGraph](https://langchain-ai.github.io/langgraph/) graphs, you do not have to rewrite them. `graph_workload` turns a compiled graph into a Draco workload. LangGraph stays the graph engine; Draco adds identity, policy, budgets, deadlines, cancellation, tracing, and interrupt-and-resume around it.

```bash
pip install "draco-harness[agents]"
```

## Run a graph

```python
from draco import Harness
from draco.adapters.langgraph import graph_workload

h = Harness()
h.register("support_graph", graph_workload(graph))
result = await h.run("support_graph", {"question": "..."}, identity=identity)
```

`graph` is any compiled graph that has `ainvoke`. Options:

| Option | Meaning |
|---|---|
| `to_state` | Turns the run's input into the graph's input state. Default: pass it through. |
| `from_state` | Turns the graph's final state into the run's output. Default: pass it through. |
| `recursion_limit` | Maximum graph supersteps (default 25). |

## What the adapter maps

| LangGraph | Draco |
|---|---|
| Thread id | The **execution id**. A resumed run keeps its id, so it continues the same graph thread. |
| Tenant, user, session and correlation ids | Passed as run metadata, **for tracing only**. They are never read back from graph state, so a graph cannot use them to impersonate anyone. |
| `GraphRecursionError` | A `budget_exhausted` result with resource `graph_steps` |
| `__interrupt__` in the result | A `waiting` result whose output is the interrupt values |
| Cancellation and the deadline | Enforced around the `ainvoke` task, effective at the graph's `await` points |

## Pause for a human, then resume

A graph node calls `interrupt(...)` to ask for input. The run becomes `waiting`; `resume` continues it by passing the answer as a `Command`:

<!-- doctest requires=langgraph -->
```python
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
    h.register("approval_flow", graph_workload(
        graph, to_state=lambda x: x if isinstance(x, Command) else {"request": x}))

    first = await h.run("approval_flow", "refund order 17")
    print("status:", first.status.value, "| waiting on:", first.output)

    done = await h.resume(first.execution_id, input=Command(resume="yes"))
    print("status:", done.status.value, "| result:", done.output)
    print("same execution id:", done.execution_id == first.execution_id)


asyncio.run(main())
```

<!-- output -->
```text
status: waiting | waiting on: [{'approve': 'refund order 17'}]
status: completed | result: {'request': 'refund order 17', 'approved': 'yes'}
same execution id: True
```

The `to_state` function handles both inputs: the first call passes a string, and the resume passes a `Command`, which must reach the graph untouched.

## Bound a runaway graph

A graph that loops forever is stopped by `recursion_limit`, and the result says which budget ran out:

<!-- doctest requires=langgraph -->
```python
import asyncio

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from draco import Harness
from draco.adapters.langgraph import graph_workload


class State(TypedDict, total=False):
    n: int


def bump(state: State) -> State:
    return {"n": state.get("n", 0) + 1}


def keep_going(state: State):
    return "bump" if state["n"] < 100 else END


builder = StateGraph(State)
builder.add_node("bump", bump)
builder.add_edge(START, "bump")
builder.add_conditional_edges("bump", keep_going)
graph = builder.compile(checkpointer=InMemorySaver())


async def main():
    h = Harness()
    h.register("loop", graph_workload(graph, recursion_limit=5))
    r = await h.run("loop", {"n": 0})
    print(r.status.value, r.error.details)


asyncio.run(main())
```

<!-- output -->
```text
budget_exhausted {'resource': 'graph_steps', 'limit': 5}
```

## Survive a restart

`InMemorySaver` forgets everything when the process exits. For a graph that waits on a human for hours, persist **both** halves of the state, or a restart loses the run:

```python
from draco.adapters.langgraph_checkpoint import build_sqlite_checkpointer
from draco.state import SqliteStore

saver = build_sqlite_checkpointer("graph-checkpoints.db")      # LangGraph's own state
graph = builder.compile(checkpointer=saver)
h = Harness(store=SqliteStore("draco-state.db"))                # Draco's execution records
```

- The **Draco store** remembers that the execution exists and is `waiting`.
- The **LangGraph checkpointer** remembers where the graph was.

Close `saver.conn` when the graph's lifecycle ends. The generic saver has no expiry policy of its own; apply your own retention if the data is sensitive.

`resume_command("approve" | "reject")` builds a validated approval `Command` for graphs that follow the approve/reject interrupt convention, and `extract_interrupts(result)` returns interrupt values without LangGraph's wrapper objects.

## Limits to know

- **Model calls inside a graph are only counted if the nodes use `call_model`.** The adapter sees a graph as one opaque call and cannot see the model calls within it. If you want `max_model_calls` and `max_tokens` to apply, have your nodes call models through [`generate` or `call_model`](model-providers.md).
- **Tool calls inside a graph are only governed if they go through `ToolRuntime`.** A node that calls a function directly bypasses the policy.
- Cancellation takes effect at the graph's `await` points. A node stuck in blocking synchronous code cannot be interrupted.

## See also

- [State, steps and resume](../concepts/state.md)
- [`draco.adapters.langgraph`](../reference/adapters.md) in the API reference
