import asyncio

import pytest

from draco.core import ContractError, DracoSettings, PolicyViolation, Status
from draco.harness import Harness, step
from draco.harness.runner import scope_key
from draco.plugins import Plugin, PluginHost
from draco.state import ExecutionRecord, SqliteStore


async def test_resume_after_process_restart_from_durable_state(tmp_path):
    """Simulate a crash: a record is left `running` with one completed step; a fresh Harness on the same
    database resumes it, skips the completed step and finishes."""
    db = str(tmp_path / "s.db")
    s1 = SqliteStore(db)
    await s1.save_execution(ExecutionRecord("exec-1", "wf", "running", 1.0, session_id="S", agent_id="wf",
                                            correlation_id="c1", input={"n": 2}))
    await s1.begin_step("exec-1", "reserve")
    await s1.complete_step("exec-1", "reserve", {"hold": "H1"})
    s1.close()  # process dies

    effects = []
    h = Harness(DracoSettings(), store=SqliteStore(db))

    async def wf(ctx, inp):
        async def reserve():
            effects.append("reserve")  # must NOT run again
            return {"hold": "H2"}
        hold = await step(ctx, "reserve", reserve)

        async def ship():
            effects.append("ship")
            return "shipped"
        return {"hold": hold["hold"], "ship": await step(ctx, "ship", ship), "n": inp["n"], "resumed": ctx.metadata["resumed"]}
    h.register("wf", wf)
    r = await h.resume("exec-1")
    assert r.status == Status.completed and r.output == {"hold": "H1", "ship": "shipped", "n": 2, "resumed": True}
    assert effects == ["ship"] and r.session_id == "S" and r.correlation_id == "c1"
    events = [e["name"] for e in await h.store.list_events("exec-1")]
    assert "execution.resumed" in events and "step.replayed" in events


async def test_interrupted_run_blocks_same_idempotency_key_until_resumed(tmp_path):
    store = SqliteStore(str(tmp_path / "s.db"))
    await store.save_execution(ExecutionRecord("e1", "w", "running", 1.0, idempotency_key=scope_key("K", None), input=1))
    h = Harness(DracoSettings(), store=store)
    ran = []

    async def w(ctx, x):
        ran.append(x)
        return "ok"
    h.register("w", w)
    r = await h.run("w", 1, idempotency_key="K")
    assert r.status == Status.failed and r.error.code == "contract" and ran == []  # no blind duplicate run
    assert (await h.resume("e1")).status == Status.completed and ran == [1]


async def test_resume_rejects_unknown_completed_and_inflight_executions():
    h = Harness(DracoSettings())
    gate = asyncio.Event()

    async def w(ctx, x):
        await gate.wait()
        return 1
    h.register("w", w)
    with pytest.raises(ContractError):
        await h.resume("missing")
    t = asyncio.create_task(h.run("w"))
    await asyncio.sleep(0.02)
    (eid,) = list(h._inflight)
    with pytest.raises(ContractError):
        await h.resume(eid)  # still running here: no double execution
    gate.set()
    await t
    with pytest.raises(ContractError):
        await h.resume(eid)  # completed


class PolicyPlugin(Plugin):
    id = "tenant-policy"
    provides = ("tenant_policy",)

    async def setup(self, ctx):
        ctx.provide("tenant_policy", {"blocked": {"acme"}})

        def check(e):
            if e.data.get("target") == "w" and ctx.get("tenant_policy")["blocked"]:
                raise PolicyViolation("tenant blocked")
        ctx.on("execution.started", check, critical=True)


async def test_plugin_installed_policy_hook_enforced_and_removed_on_dispose():
    host = PluginHost()
    h = Harness(DracoSettings(), host=host)
    h.register("w", lambda ctx, x: asyncio.sleep(0, "ran"))
    host.register(PolicyPlugin())
    await host.start()
    assert (await h.run("w")).error.code == "policy"  # extension changed behaviour without touching core
    await host.stop()
    assert (await h.run("w")).output == "ran"  # hook gone after disposal; services released
    assert not host.services.has("tenant_policy") and host.services.has("events")  # protected core services stay


async def test_plugin_cannot_override_core_services():
    host = PluginHost()
    Harness(DracoSettings(), host=host)

    class Evil(Plugin):
        id = "evil"

        async def setup(self, ctx):
            ctx.replace("store", object())
    host.register(Evil())
    from draco.core import PluginError
    with pytest.raises(PluginError):
        await host.start()


async def test_sqlite_state_isolated_between_concurrent_sessions(tmp_path):
    h = Harness(DracoSettings(), store=SqliteStore(str(tmp_path / "s.db")))

    async def w(ctx, x):
        await step(ctx, "a", lambda: asyncio.sleep(0, {"who": x}))
        await asyncio.sleep(0.005)
        return (await h.store.get_step(ctx.execution_id, "a")).data["who"]
    h.register("w", w)
    rs = await asyncio.gather(*(h.run("w", f"u{i}", session_id=f"s{i}") for i in range(8)))
    assert [r.output for r in rs] == [f"u{i}" for i in range(8)]
    assert len({r.execution_id for r in rs}) == 8
