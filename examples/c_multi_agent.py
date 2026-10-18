"""Example C: route a request, hand off between agents, delegate, and print the parent/child trace."""
import asyncio
import json

from draco import Handoff, Harness
from draco.agents import AgentRegistry, Router, RouteRequest


async def main():
    h = Harness()
    agents = AgentRegistry(h)
    router = Router(h, agents)

    async def triage(ctx, text):
        # decide in code; a handoff is validated by the router before the destination runs
        return Handoff("math", "arithmetic detected", {"a": 2, "b": 3}) if "+" in text else Handoff("writer", "prose")

    async def math(ctx, x):
        return {"sum": x["a"] + x["b"], "checked_by": await router.delegate(ctx, "auditor", x)}

    async def auditor(ctx, x):
        return "auditor confirms inputs are small integers"

    async def writer(ctx, x):
        return {"text": "..."}

    agents.register("triage", triage, capabilities={"triage"})
    agents.register("math", math, capabilities={"calc"})
    agents.register("auditor", auditor)
    agents.register("writer", writer, capabilities={"write"})

    r = await router.run(RouteRequest("what is 2+3 ?", agent="triage"), session_id="s1")
    print("status:", r.status.value)
    print("path:", r.output["path"], "| strategies:", [d["strategy"] for d in r.output["decisions"]])
    print("answer:", json.dumps(r.output["output"]))
    print("execution tree:")

    def show(node, indent=0):
        print("  " * indent + f"- {node['agent_id']} [{node['status']}] {node['execution_id'][:8]}")
        for c in node["children"]:
            show(c, indent + 1)
    show(h.tracer.tree(r.execution_id))

    bad = await router.run(RouteRequest("x", agent="does-not-exist"))
    print("unknown agent ->", bad.status.value, bad.error.code, "|", bad.error.message)

asyncio.run(main())
