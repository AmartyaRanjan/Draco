# Route between agents

Draco treats an **agent** as a workload with a name and some labels. This guide shows how to register several agents, route a request to the right one, let one agent hand off to another, and let one call another as a helper. Every destination is checked before it runs.

## Agents and the router

`AgentRegistry` registers agents (they are ordinary harness workloads, plus routing metadata). `Router` chooses one for a request and follows its handoffs:

```python
from draco import Harness
from draco.agents import AgentRegistry, RouteRequest, Router

h = Harness()
agents = AgentRegistry(h)
router = Router(h, agents)

agents.register("billing", billing, capabilities={"money"})
agents.register("support", support, capabilities={"help"})

result = await router.run(RouteRequest("where is my refund?"))
```

`router.run(...)` returns an ordinary `ExecutionResult`. On success its `output` is a dict with:

| Key | Contents |
|---|---|
| `output` | What the final agent returned |
| `path` | The agents that ran, in order, such as `['triage', 'billing']` |
| `decisions` | Why each hop happened: destination, reason, and which strategy decided |
| `status`, `final_execution_id` | How the last agent ended, and its execution id |

## Strategies

A router tries its **strategies** in order, and the first one that returns a decision wins:

| Strategy | Picks the agent when... |
|---|---|
| `explicit()` | The request names one: `RouteRequest(input, agent="billing")` |
| `rules((predicate, agent, reason), ...)` | A predicate on the request is true |
| `by_capability()` | The request lists `require={...}` capabilities; the agent with the best match (fewest extras) wins |
| `llm_proposal(fn)` | Your function, usually a model call, suggests an agent. Optional, and treated as a **suggestion** that is validated like any other. |

The default is `[explicit(), by_capability()]`. If no strategy decides, the router uses `default_agent` if you set one, and otherwise fails with a `routing` error ("no route found"). Draco never needs a model to route; a deterministic rule is often better.

## Handoffs and delegation

An agent can **hand off** by returning a `Handoff(destination, reason, input)` instead of an answer. The router validates the destination, then runs it. This is how a triage agent passes work along.

An agent can also **delegate**: `await router.delegate(ctx, "auditor", data)` runs another agent as a child of the current run and returns its output. A failure in the child is re-raised as its original typed error. Use delegation for a helper whose answer you need, and a handoff to pass control and be done.

## Safety rules

- **Every destination is validated before it runs**: a name in a handoff, a model's suggestion, an explicit request. An unknown name is a `routing` error and never executes.
- **The policy's `allowed_agents` is enforced**: pass `Policy(allowed_agents=frozenset({...}))` and any other destination is denied, whichever strategy chose it.
- **Handoff loops are caught.** Going back to an agent already on the path raises `RoutingError`, unless you create the router with `allow_revisit=True`. More than `max_handoffs` hops (default 5) also raises it.
- **Runs nest under one execution tree**, sharing the budget and identity, so an agent cannot get more resources by calling more agents.

## Example

<!-- doctest -->
```python
import asyncio
from draco import Handoff, Harness, Policy
from draco.agents import AgentRegistry, RouteRequest, Router, by_capability, explicit, rules

async def main():
    h = Harness()
    agents = AgentRegistry(h)

    async def billing(ctx, text):
        return f"billing handled: {text}"

    async def support(ctx, text):
        return f"support handled: {text}"

    async def triage(ctx, text):
        if "invoice" in text:
            return Handoff("billing", "mentions an invoice", text)
        return Handoff("support", "default")

    async def auditor(ctx, data):
        return "audited"

    async def lead(ctx, data):
        return {"answer": data, "audit": await router.delegate(ctx, "auditor", data)}

    agents.register("billing", billing, capabilities={"money"})
    agents.register("support", support, capabilities={"help"})
    agents.register("triage", triage, capabilities={"triage"})
    agents.register("auditor", auditor)
    agents.register("lead", lead, capabilities={"lead"})

    router = Router(
        h, agents,
        strategies=[explicit(), rules((lambda req: "refund" in str(req.input), "billing", "refund keyword")), by_capability()],
        default_agent="support",
    )

    def show(label, r):
        if r.ok:
            print(f"{label}: path={r.output['path']} strategies={[d['strategy'] for d in r.output['decisions']]}")
        else:
            print(f"{label}: {r.status.value} {r.error.code}: {r.error.message}")

    show("rule", await router.run(RouteRequest("where is my refund")))
    show("capability", await router.run(RouteRequest("hello", require=frozenset({"help"}))))
    show("default", await router.run(RouteRequest("hello")))
    show("handoff", await router.run(RouteRequest("invoice 7 is wrong", agent="triage")))
    show("unknown agent", await router.run(RouteRequest("x", agent="ghost")))
    show("not allowed", await router.run(RouteRequest("x", agent="billing"),
                                         policy=Policy(allowed_agents=frozenset({"support", "router"}))))

    r = await router.run(RouteRequest("x", agent="lead"))
    print("delegate:", r.output["output"])

    def tree(node, depth=0):
        print("  " * depth + f"{node['agent_id']} [{node['status']}]")
        for child in node["children"]:
            tree(child, depth + 1)

    tree(h.tracer.tree(r.execution_id))

asyncio.run(main())
```

<!-- output -->
```text
rule: path=['billing'] strategies=['rule']
capability: path=['support'] strategies=['capability']
default: path=['support'] strategies=['default']
handoff: path=['triage', 'billing'] strategies=['explicit', 'handoff']
unknown agent: failed routing: unknown agent 'ghost'
not allowed: failed policy: agent 'billing' is not allowed for this execution
delegate: {'answer': 'x', 'audit': 'audited'}
router [completed]
  lead [completed]
    auditor [completed]
```

Two things to notice in the output. The strategy names in `decisions` are `rule`, `capability`, `default`, `explicit` and `handoff`, so you can see *why* each route was chosen. And `h.tracer.tree(...)` shows the whole run as a tree: the router, the agent it chose, and the helper that agent delegated to.

## See also

- [Context, results and errors](../concepts/execution-model.md) for how nested runs share a budget and an identity.
- [`RoutingError`](../reference/errors.md) for the failures a router raises.
