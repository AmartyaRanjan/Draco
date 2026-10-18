# Extending Draco

Everything here works without editing Draco's source. See `examples/f_plugin.py` (runs in CI as a test).

## Plugin
```python
from pydantic import BaseModel
from draco.plugins import Plugin, PluginHost
from draco.core import PolicyViolation

class Cfg(BaseModel):
    limit: int = 2

class Quota(Plugin):
    id = "quota"                       # unique
    provides = ("quota",)              # service names it registers
    requires = ()                      # service names it needs (provided by plugins or core)
    config_model = Cfg                 # validated from PluginHost(config={"quota": {...}})

    async def setup(self, ctx):        # register services and hooks
        used = {}
        ctx.provide("quota", used)
        def enforce(e):
            if used.setdefault(e.data["user_id"], 0) >= ctx.config.limit: raise PolicyViolation("quota")
            used[e.data["user_id"]] += 1
        ctx.on("execution.started", enforce, critical=True)   # fail-closed policy hook
    async def start(self, ctx): ...    # open connections
    async def stop(self, ctx): ...     # close them (reverse start order)
    async def dispose(self, ctx): ...  # final cleanup; services and hooks are removed automatically

host = PluginHost(config={"quota": {"limit": 5}})
h = Harness(host=host)
host.register(Quota()); await host.start(); ...; await host.stop()
```
Semantics: providers start before dependents; cycles, duplicate plugin ids, duplicate service names and missing dependencies are errors **before anything runs**; if `setup`/`start` fails, already-started plugins are stopped and disposed in reverse order and a `PluginError` (cause attached) is raised. Core services (`events`, `settings`, `store`, `tracer`, `harness`) are protected: plugins cannot replace them. A plugin may only replace a service it owns.

## Which hook semantics to pick
- Observing (metrics, audit export): `ctx.on(event, fn)`; errors are isolated and recorded.
- Enforcing (quota, tenant block, approval): `critical=True`; raising `PolicyViolation` stops the run or the tool call. Critical hooks run before non-critical ones.

## Your own workload, tool, router strategy, store, exporter
- Workload/agent: any `async def (ctx, input)`; use `call_model`, `step`, `ctx.check()` and `h.run(..., parent=ctx)` for nested work.
- Tool: `native_tool(...)`; raise `CapabilityError(ErrorCode.x, msg, retryable=...)` for expected failures.
- Router strategy: a callable `(ctx, RouteRequest, registry) -> RoutingDecision | None`.
- Store: implement `ExecutionStore`, `EventStore`, `CheckpointStore` (`draco.state.stores`) and pass `Harness(store=...)`.
- Exporter: an object with `export(record: dict)`; pass `Harness(exporters=[...])`.
