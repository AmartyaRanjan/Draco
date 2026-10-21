"""Example F: extend Draco with a plugin that provides a service and installs hooks, without
modifying Draco's source. Shows dependency ordering, config injection and clean disposal."""
import asyncio

from pydantic import BaseModel

from draco import DracoSettings, Harness
from draco.core import PolicyViolation
from draco.plugins import Plugin, PluginHost


class QuotaConfig(BaseModel):
    max_runs_per_user: int = 2


class QuotaPlugin(Plugin):
    id = "quota"
    provides = ("quota",)
    config_model = QuotaConfig

    async def setup(self, ctx):
        used: dict[str, int] = {}
        ctx.provide("quota", used)

        def enforce(event):  # critical hook: fails closed
            user = event.data.get("user_id") or "anonymous"
            used[user] = used.get(user, 0) + 1
            if used[user] > ctx.config.max_runs_per_user:
                raise PolicyViolation(f"quota exceeded for {user}")
        ctx.on("execution.started", enforce, critical=True)


class AuditPlugin(Plugin):
    id = "audit"
    requires = ("quota",)  # starts after quota

    async def setup(self, ctx):
        self.lines = ctx.get("quota")
        ctx.on("execution.finished", lambda e: print("   audit:", e.data["status"], e.data["target"]))


async def main():
    host = PluginHost(config={"quota": {"max_runs_per_user": 2}})
    h = Harness(DracoSettings(), host=host)

    async def work(ctx, x):
        return x
    h.register("work", work)
    host.register(AuditPlugin())  # registered first, started second: order comes from `requires`
    host.register(QuotaPlugin())
    await host.start()
    print("started in order:", host.started, "| services:", host.services.names())
    for i in range(3):
        r = await h.run("work", i, user_id="alice")
        print(f"run {i}:", r.status.value, r.error.message if r.error else "")
    await host.stop()
    print("after stop: quota service present?", host.services.has("quota"))
    print("run after plugins disposed:", (await h.run("work", 9)).status.value)

asyncio.run(main())
