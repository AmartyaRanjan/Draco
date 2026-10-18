import asyncio

from draco.agents import AgentRegistry, Router, RouteRequest, llm_proposal, rules
from draco.core import BudgetLimits, DracoSettings, Handoff, Policy, Status
from draco.harness import Harness


def setup(**router_kw):
    h = Harness(DracoSettings(default_timeout_s=5))
    reg = AgentRegistry(h)

    async def math(ctx, x):
        return {"answer": x["a"] + x["b"], "by": "math"}

    async def writer(ctx, x):
        return {"text": f"wrote about {x}", "by": "writer"}

    async def triage(ctx, x):
        return Handoff("math", "numbers detected", {"a": 2, "b": 3})
    reg.register("math", math, capabilities={"calc"})
    reg.register("writer", writer, capabilities={"write", "summarize"})
    reg.register("triage", triage, capabilities={"triage"})
    return h, reg, Router(h, reg, **router_kw)


async def test_explicit_rule_and_capability_routing():
    h, reg, router = setup(default_agent="writer")
    r = await router.run(RouteRequest({"a": 1, "b": 1}, agent="math"))
    assert r.output["path"] == ["math"] and r.output["output"]["answer"] == 2
    r = await router.run(RouteRequest("x", require=frozenset({"summarize"})))
    assert r.output["path"] == ["writer"] and r.output["decisions"][0]["strategy"] == "capability"
    r = await router.run("anything")  # falls back to default
    assert r.output["decisions"][0]["strategy"] == "default"
    h2, reg2, router2 = setup()
    router2.strategies = [rules((lambda q: "sum" in str(q.input), "math", "arithmetic"))]
    r = await router2.run(RouteRequest({"a": 1, "b": 2, "q": "sum"}))
    assert r.output["output"]["answer"] == 3 and r.output["decisions"][0]["strategy"] == "rule"


async def test_unknown_and_policy_denied_destinations_never_execute():
    h, reg, router = setup()
    r = await router.run(RouteRequest("x", agent="ghost"))
    assert r.status == Status.failed and r.error.code == "routing" and "ghost" in r.error.message
    r = await router.run(RouteRequest("x"))  # no route, no default
    assert r.error.code == "routing"
    r = await router.run(RouteRequest({"a": 1, "b": 1}, agent="math"), policy=Policy(allowed_agents=frozenset({"router", "writer"})))
    assert r.error.code == "policy"
    assert h.tracer.trace(r.execution_id)  # router execution itself is traced


async def test_llm_proposal_is_validated_not_trusted():
    h, reg, router = setup()
    router.strategies = [llm_proposal(lambda inp, agents: "rm_rf_everything")]
    r = await router.run("x")
    assert r.error.code == "routing"

    async def good(inp, agents):
        return {"destination": "writer", "reason": "prose", "input": "topic"}
    router.strategies = [llm_proposal(good)]
    r = await router.run("x")
    assert r.output["output"]["text"] == "wrote about topic"
    router.strategies = [llm_proposal(lambda i, a: {"nope": 1})]
    assert (await router.run("x")).error.code == "routing"


async def test_handoff_chain_parent_child_tracing_and_termination():
    h, reg, router = setup()
    r = await router.run(RouteRequest("2+3", agent="triage"), session_id="S")
    assert r.status == Status.completed
    assert r.output["path"] == ["triage", "math"] and r.output["output"]["answer"] == 5
    assert [d["strategy"] for d in r.output["decisions"]] == ["explicit", "handoff"]
    tree = h.tracer.tree(r.execution_id)
    assert tree["agent_id"] == "router" and [c["agent_id"] for c in tree["children"]] == ["triage", "math"]
    assert tree["children"][0]["status"] == "handoff" and tree["children"][1]["status"] == "completed"
    assert len(r.children) == 2
    assert "handoff" in [e["event"] for c in tree["children"] for e in h.tracer.trace(c["execution_id"])]


async def test_handoff_loop_limit_and_invalid_handoff_destination():
    h, reg, router = setup(max_handoffs=3)

    async def ping(ctx, x):
        return Handoff("pong")

    async def pong(ctx, x):
        return Handoff("ping")

    async def bad(ctx, x):
        return Handoff("does-not-exist")
    reg.register("ping", ping)
    reg.register("pong", pong)
    reg.register("bad", bad)
    r = await router.run(RouteRequest("x", agent="ping"))
    assert r.error.code == "routing" and "loop" in r.error.message
    r = await router.run(RouteRequest("x", agent="bad"))
    assert r.error.code == "routing" and "does-not-exist" in r.error.message

    h2, reg2, router2 = setup(max_handoffs=1, allow_revisit=True)
    reg2.register("ping", ping); reg2.register("pong", pong)
    r = await router2.run(RouteRequest("x", agent="ping"))
    assert "limit" in r.error.message


async def test_delegation_propagates_failure_cancellation_and_budget():
    h, reg, router = setup()

    async def boss(ctx, x):
        return await router.delegate(ctx, "math", {"a": 4, "b": 4})

    async def broken(ctx, x):
        raise ValueError("kaput")

    async def boss2(ctx, x):
        return await router.delegate(ctx, "broken", x)
    reg.register("boss", boss)
    reg.register("broken", broken)
    reg.register("boss2", boss2)
    r = await h.run("boss")
    assert r.output["answer"] == 8 and len(r.children) == 1
    r = await h.run("boss2")
    assert r.status == Status.failed  # child failure surfaces to the parent as an error
    tree = h.tracer.tree(r.execution_id)
    assert tree["children"][0]["status"] == "failed"

    async def spend(ctx, x):
        from draco.harness import call_model

        async def m():
            return 1
        await call_model(ctx, m)
        return 1

    async def parent(ctx, x):
        for _ in range(3):
            await router.delegate(ctx, "spend", x)
    reg.register("spend", spend)
    reg.register("parent", parent)
    r = await h.run("parent", limits=BudgetLimits(max_model_calls=2))
    assert r.status == Status.budget_exhausted  # child spend counts against the shared budget


async def test_cancellation_reaches_nested_agents():
    h, reg, router = setup()
    stopped = []

    async def slow(ctx, x):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            stopped.append("slow")
            raise
    reg.register("slow", slow)
    t = asyncio.create_task(router.run(RouteRequest("x", agent="slow")))
    await asyncio.sleep(0.05)
    for eid in list(h._inflight):
        if h._inflight[eid].parent_execution_id is None:
            h.cancel(eid)
    r = await t
    assert r.status == Status.cancelled and stopped == ["slow"]
