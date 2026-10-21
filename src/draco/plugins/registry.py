"""Plugin host: dependency ordering, lifecycle and deterministic cleanup.

Plugins declare the services they `provide` and `require`. The host orders them so providers run
first, runs `setup` then `start` per plugin, and on any failure stops and disposes everything it
already brought up, in reverse order, before raising.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from draco.core.errors import (
    ConfigurationError,
    DependencyCycle,
    DuplicateRegistration,
    MissingDependency,
    PluginError,
)
from draco.core.events import Event, EventBus, Handler
from draco.plugins.services import ServiceRegistry

logger = logging.getLogger("draco.plugins")


class PluginContext:
    """What a plugin may touch: its own config, service registration and event subscriptions.
    Everything registered here is tracked so dispose can undo it."""

    def __init__(self, plugin_id: str, config: Any, services: ServiceRegistry, bus: EventBus):
        self.plugin_id, self.config = plugin_id, config
        self._services, self._bus = services, bus
        self._unsubs: list[Callable[[], None]] = []

    def provide(self, name: str, service: Any) -> None:
        self._services.provide(name, service, owner=self.plugin_id)

    def replace(self, name: str, service: Any) -> Any:
        return self._services.replace(name, service, owner=self.plugin_id)

    def get(self, name: str) -> Any:
        return self._services.get(name)

    def on(self, pattern: str, handler: Handler, *, critical: bool = False) -> None:
        self._unsubs.append(self._bus.subscribe(pattern, handler, critical=critical))

    async def emit(self, name: str, **data: Any) -> None:
        await self._bus.emit(Event(name, data=data))

    def _release(self) -> None:
        for u in reversed(self._unsubs):
            u()
        self._unsubs.clear()
        self._services.remove_owned_by(self.plugin_id)


class Plugin:
    """Base class; override what you need. `id` must be unique."""
    id: str = ""
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    config_model: type[BaseModel] | None = None

    async def setup(self, ctx: PluginContext) -> None: ...
    async def start(self, ctx: PluginContext) -> None: ...
    async def stop(self, ctx: PluginContext) -> None: ...
    async def dispose(self, ctx: PluginContext) -> None: ...


class PluginHost:
    def __init__(self, services: ServiceRegistry | None = None, bus: EventBus | None = None,
                 config: dict[str, Any] | None = None):
        self.services = services or ServiceRegistry()
        self.bus = bus or EventBus()
        self._config = config or {}
        self._plugins: dict[str, Plugin] = {}
        self._ctx: dict[str, PluginContext] = {}
        self._started: list[str] = []  # ids in start order
        self._running = False

    def register(self, plugin: Plugin) -> None:
        if not plugin.id:
            raise PluginError("plugin must declare a non-empty id")
        if plugin.id in self._plugins:
            raise DuplicateRegistration(f"plugin '{plugin.id}' already registered")
        if self._running:
            raise PluginError("cannot register plugins after the host started")
        for other in self._plugins.values():
            clash = set(other.provides) & set(plugin.provides)
            if clash:
                raise DuplicateRegistration(
                    f"service(s) {sorted(clash)} already provided by plugin '{other.id}'")
        self._plugins[plugin.id] = plugin

    def order(self) -> list[Plugin]:
        provider = {s: p.id for p in self._plugins.values() for s in p.provides}
        deps: dict[str, set[str]] = {}
        for p in self._plugins.values():
            d: set[str] = set()
            for need in p.requires:
                if need in provider:
                    d.add(provider[need])
                elif not self.services.has(need):
                    raise MissingDependency(f"plugin '{p.id}' requires service '{need}', which nothing provides")
            deps[p.id] = d
        out: list[Plugin] = []
        state: dict[str, int] = {}

        def visit(pid: str, trail: list[str]) -> None:
            if state.get(pid) == 2:
                return
            if state.get(pid) == 1:
                cyc = trail[trail.index(pid):] + [pid]
                raise DependencyCycle("plugin dependency cycle: " + " -> ".join(cyc))
            state[pid] = 1
            for d in sorted(deps[pid]):
                visit(d, trail + [pid])
            state[pid] = 2
            out.append(self._plugins[pid])

        for pid in self._plugins:  # registration order breaks ties deterministically
            visit(pid, [])
        return out

    def _make_ctx(self, p: Plugin) -> PluginContext:
        cfg = self._config.get(p.id, {})
        if p.config_model is not None:
            try:
                cfg = p.config_model.model_validate(cfg)
            except ValidationError as exc:
                raise ConfigurationError(f"invalid config for plugin '{p.id}': {exc.error_count()} error(s)") from exc
        return PluginContext(p.id, cfg, self.services, self.bus)

    async def start(self) -> None:
        ordered = self.order()  # validates before anything runs
        ctxs = {p.id: self._make_ctx(p) for p in ordered}
        self._ctx = ctxs
        self._running = True
        for p in ordered:
            ctx = ctxs[p.id]
            stage = "setup"
            try:
                await p.setup(ctx)
                stage = "start"
                await p.start(ctx)
            except Exception as exc:
                await self.bus.emit(Event("plugin.failed", data={"plugin": p.id, "stage": stage}))
                if stage == "start":  # setup succeeded: undo it for this plugin as well
                    await self._dispose_one(p)
                else:
                    await self._dispose_one(p)
                await self._unwind()
                raise PluginError(f"plugin '{p.id}' failed during {stage}: {exc}",
                                  details={"plugin": p.id, "stage": stage}) from exc
            self._started.append(p.id)
            await self.bus.emit(Event("plugin.started", data={"plugin": p.id}))

    async def _dispose_one(self, p: Plugin) -> None:
        ctx = self._ctx[p.id]
        try:
            await p.dispose(ctx)
        except Exception:
            logger.warning("plugin_dispose_failed", extra={"plugin": p.id})
        ctx._release()

    async def _unwind(self) -> list[Exception]:
        errors: list[Exception] = []
        for pid in reversed(self._started):
            p, ctx = self._plugins[pid], self._ctx[pid]
            try:
                await p.stop(ctx)
            except Exception as exc:
                errors.append(exc)
                logger.warning("plugin_stop_failed", extra={"plugin": pid})
            await self._dispose_one(p)
            await self.bus.emit(Event("plugin.stopped", data={"plugin": pid}))
        self._started.clear()
        self._running = False
        return errors

    async def stop(self) -> None:
        """Stop and dispose in reverse start order. Always finishes; raises the first stop error after."""
        errors = await self._unwind()
        if errors:
            raise PluginError(f"{len(errors)} plugin(s) failed to stop cleanly: {errors[0]}") from errors[0]

    async def __aenter__(self) -> PluginHost:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    @property
    def started(self) -> list[str]:
        return list(self._started)
