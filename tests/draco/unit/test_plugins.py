import pytest

from draco.core.errors import (
    ConfigurationError,
    DependencyCycle,
    DuplicateRegistration,
    MissingDependency,
    PluginError,
    PolicyViolation,
)
from draco.plugins import Plugin, PluginHost, ServiceRegistry


class P(Plugin):
    def __init__(self, id, provides=(), requires=(), log=None, fail=None):
        self.id, self.provides, self.requires = id, tuple(provides), tuple(requires)
        self.log, self.fail = log if log is not None else [], fail

    async def setup(self, ctx):
        self.log.append(f"setup:{self.id}")
        if self.fail == "setup":
            raise RuntimeError("boom")
        for s in self.provides:
            ctx.provide(s, object())

    async def start(self, ctx):
        self.log.append(f"start:{self.id}")
        if self.fail == "start":
            raise RuntimeError("boom")

    async def stop(self, ctx):
        self.log.append(f"stop:{self.id}")

    async def dispose(self, ctx):
        self.log.append(f"dispose:{self.id}")


async def test_dependency_order_and_reverse_shutdown():
    log = []
    host = PluginHost()
    host.register(P("c", requires=["b"], log=log))
    host.register(P("b", provides=["b"], requires=["a"], log=log))
    host.register(P("a", provides=["a"], log=log))
    await host.start()
    assert host.started == ["a", "b", "c"]
    await host.stop()
    assert [x for x in log if x.startswith("stop")] == ["stop:c", "stop:b", "stop:a"]
    assert not host.services.has("a")  # services released on dispose


def test_cycle_detected():
    host = PluginHost()
    host.register(P("a", provides=["a"], requires=["b"]))
    host.register(P("b", provides=["b"], requires=["a"]))
    with pytest.raises(DependencyCycle):
        host.order()


def test_missing_dependency_and_duplicates():
    host = PluginHost()
    host.register(P("a", requires=["nope"]))
    with pytest.raises(MissingDependency):
        host.order()
    with pytest.raises(DuplicateRegistration):
        host.register(P("a"))
    host.register(P("x", provides=["svc"]))
    with pytest.raises(DuplicateRegistration):
        host.register(P("y", provides=["svc"]))


async def test_startup_failure_rolls_back_in_reverse():
    log = []
    host = PluginHost()
    host.register(P("a", provides=["a"], log=log))
    host.register(P("b", provides=["b"], requires=["a"], log=log, fail="start"))
    with pytest.raises(PluginError) as ei:
        await host.start()
    assert isinstance(ei.value.__cause__, RuntimeError)
    assert log == ["setup:a", "start:a", "setup:b", "start:b", "dispose:b", "stop:a", "dispose:a"]
    assert host.started == [] and host.services.names() == []


async def test_setup_failure_disposes_failing_plugin():
    log = []
    host = PluginHost()
    host.register(P("a", log=log, fail="setup"))
    with pytest.raises(PluginError):
        await host.start()
    assert "dispose:a" in log and "start:a" not in log


async def test_protected_service_cannot_be_replaced_by_plugin():
    reg = ServiceRegistry()
    reg.provide("policy", object(), protected=True)

    class Evil(Plugin):
        id = "evil"

        async def setup(self, ctx):
            ctx.replace("policy", object())

    host = PluginHost(services=reg)
    host.register(Evil())
    with pytest.raises(PluginError) as ei:
        await host.start()
    assert isinstance(ei.value.__cause__, PolicyViolation)


async def test_plugin_hook_and_config_validation():
    from pydantic import BaseModel

    class Cfg(BaseModel):
        n: int

    seen = []

    class H(Plugin):
        id, config_model = "h", Cfg

        async def setup(self, ctx):
            ctx.on("x.*", lambda e: seen.append((e.name, ctx.config.n)))

    host = PluginHost(config={"h": {"n": 3}})
    host.register(H())
    async with host:
        from draco.core.events import Event
        await host.bus.emit(Event("x.y"))
    assert seen == [("x.y", 3)]
    bad = PluginHost(config={"h": {"n": "nope"}})
    bad.register(H())
    with pytest.raises(ConfigurationError):
        await bad.start()
