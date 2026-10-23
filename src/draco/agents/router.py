"""Deterministic-first routing with validated destinations, handoffs and loop limits."""
from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, cast

from draco.agents.registry import AgentRegistry, AgentSpec
from draco.core.context import ExecutionContext
from draco.core.errors import ContractError, RoutingError
from draco.core.results import ExecutionResult, Handoff, Status
from draco.harness.instrument import emit
from draco.harness.runner import Harness

ROUTER = "router"


@dataclass
class RoutingDecision:
    destination: str
    reason: str = ""
    input: Any = None  # optional structured input for the destination
    strategy: str = ""


@dataclass
class RouteRequest:
    input: Any
    agent: str | None = None  # explicit target
    require: frozenset[str] = frozenset()  # capabilities the destination must have
    hints: dict[str, Any] = field(default_factory=dict)


Strategy = Callable[[ExecutionContext, RouteRequest, AgentRegistry], Awaitable[RoutingDecision | None] | RoutingDecision | None]


def explicit() -> Strategy:
    def s(ctx: ExecutionContext, req: RouteRequest, reg: AgentRegistry) -> RoutingDecision | None:
        return RoutingDecision(req.agent, "explicit request", strategy="explicit") if req.agent else None
    return s


def rules(*rule_list: tuple[Callable[[RouteRequest], bool], str, str]) -> Strategy:
    """(predicate, destination, reason) tuples; first match wins."""
    def s(ctx: ExecutionContext, req: RouteRequest, reg: AgentRegistry) -> RoutingDecision | None:
        for pred, dest, reason in rule_list:
            if pred(req):
                return RoutingDecision(dest, reason, strategy="rule")
        return None
    return s


def by_capability() -> Strategy:
    """Pick the agent that has every required capability with the fewest extras (name breaks ties)."""
    def s(ctx: ExecutionContext, req: RouteRequest, reg: AgentRegistry) -> RoutingDecision | None:
        if not req.require:
            return None
        cands = reg.with_capabilities(req.require)
        if not cands:
            return None
        best = min(cands, key=lambda a: (len(a.capabilities - req.require), a.name))
        return RoutingDecision(best.name, f"capabilities {sorted(req.require)}", strategy="capability")
    return s


def llm_proposal(propose: Callable[[Any, list[AgentSpec]], Awaitable[Any] | Any]) -> Strategy:
    """Optional. `propose(input, agents)` returns an agent name or {"destination","reason","input"}.
    The proposal is only a suggestion: the router validates it like any other decision."""
    async def s(ctx: ExecutionContext, req: RouteRequest, reg: AgentRegistry) -> RoutingDecision | None:
        out = propose(req.input, reg.all())
        if inspect.isawaitable(out):
            out = await out
        if out is None:
            return None
        if isinstance(out, str):
            return RoutingDecision(out.strip(), "llm proposal", strategy="llm")
        if isinstance(out, dict) and isinstance(out.get("destination"), str):
            return RoutingDecision(out["destination"].strip(), str(out.get("reason", "llm proposal")),
                                   out.get("input"), "llm")
        raise RoutingError("routing proposal is malformed")
    return s


class Router:
    def __init__(self, harness: Harness, registry: AgentRegistry, strategies: list[Strategy] | None = None, *,
                 default_agent: str | None = None, max_handoffs: int = 5, allow_revisit: bool = False):
        self.harness, self.registry = harness, registry
        self.strategies = strategies or [explicit(), by_capability()]
        self.default_agent, self.max_handoffs, self.allow_revisit = default_agent, max_handoffs, allow_revisit
        harness.register(ROUTER, self._workload, description="routes a request to registered agents")

    def _validate(self, ctx: ExecutionContext, destination: str) -> AgentSpec:
        spec = self.registry.get(destination)  # unknown destination -> RoutingError, never executed
        ctx.policy.authorize_agent(destination)
        return spec

    async def decide(self, ctx: ExecutionContext, req: RouteRequest) -> RoutingDecision:
        for strat in self.strategies:
            raw = strat(ctx, req, self.registry)
            if inspect.isawaitable(raw):
                decision = await cast(Awaitable[RoutingDecision | None], raw)
            else:
                decision = cast(RoutingDecision | None, raw)
            if decision is not None:
                self._validate(ctx, decision.destination)
                return decision
        if self.default_agent:
            self._validate(ctx, self.default_agent)
            return RoutingDecision(self.default_agent, "default agent", strategy="default")
        raise RoutingError("no route found for request", details={"require": sorted(req.require)})

    async def _workload(self, ctx: ExecutionContext, req: RouteRequest) -> dict[str, Any]:
        decisions: list[RoutingDecision] = []
        path: list[str] = []
        decision = await self.decide(ctx, req)
        payload = decision.input if decision.input is not None else req.input
        while True:
            if not self.allow_revisit and decision.destination in path:
                raise RoutingError(f"handoff loop detected: {' -> '.join(path + [decision.destination])}")
            if len(path) > self.max_handoffs:
                raise RoutingError(f"handoff limit of {self.max_handoffs} exceeded")
            decisions.append(decision)
            path.append(decision.destination)
            await emit(ctx, "route.decided", destination=decision.destination, reason=decision.reason,
                       strategy=decision.strategy, hop=len(path))
            res = await self.harness.run(decision.destination, payload, parent=ctx)
            if res.status == Status.handoff:
                h: Handoff = res.handoff  # type: ignore[assignment]
                self._validate(ctx, h.destination)  # validated before it can run
                decision = RoutingDecision(h.destination, h.reason, h.input, "handoff")
                payload = h.input if h.input is not None else payload
                continue
            if res.status != Status.waiting:
                res.raise_for_status()
            return {"output": res.output, "path": path, "status": res.status.value,
                    "decisions": [d.__dict__ for d in decisions], "final_execution_id": res.execution_id}

    async def run(self, request: RouteRequest | Any, **kwargs: Any) -> ExecutionResult:
        if not isinstance(request, RouteRequest):
            request = RouteRequest(input=request)
        return await self.harness.run(ROUTER, request, **kwargs)

    async def delegate(self, ctx: ExecutionContext, destination: str, input: Any) -> Any:
        """Run another agent as a child of the current execution and return its output, raising its
        failure as a typed error. Destination is validated against the registry and policy first."""
        self._validate(ctx, destination)
        res = await self.harness.run(destination, input, parent=ctx)
        if res.status == Status.handoff:
            raise ContractError("a delegated agent returned a handoff; use Router.run for handoff chains")
        res.raise_for_status()
        return res.output


__all__ = ["Router", "RouteRequest", "RoutingDecision", "explicit", "rules", "by_capability", "llm_proposal"]
