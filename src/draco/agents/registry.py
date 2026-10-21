from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from draco.core.errors import ContractError, RoutingError
from draco.core.protocols import Workload
from draco.harness.runner import Harness


@dataclass(frozen=True)
class AgentSpec:
    name: str
    description: str = ""
    capabilities: frozenset[str] = frozenset()
    meta: dict[str, Any] = field(default_factory=dict)


class AgentRegistry:
    """Agents are harness workloads plus routing metadata. Provider-independent: an agent is any
    async callable (a LangGraph graph, a RAG pipeline, a plain function)."""

    def __init__(self, harness: Harness):
        self.harness = harness
        self._agents: dict[str, AgentSpec] = {}

    def register(self, name: str, fn: Workload, *, description: str = "",
                 capabilities: set[str] | frozenset[str] = frozenset(), **meta: Any) -> AgentSpec:
        if not name or name == "router":
            raise ContractError(f"invalid agent name '{name}'")
        if name in self._agents:
            raise ContractError(f"agent '{name}' already registered")
        self.harness.register(name, fn, description=description, capabilities=capabilities, **meta)
        spec = AgentSpec(name, description, frozenset(capabilities), meta)
        self._agents[name] = spec
        return spec

    def get(self, name: str) -> AgentSpec:
        try:
            return self._agents[name]
        except KeyError:
            raise RoutingError(f"unknown agent '{name}'", details={"known": sorted(self._agents)}) from None

    def has(self, name: str) -> bool:
        return name in self._agents

    def all(self) -> list[AgentSpec]:
        return [self._agents[k] for k in sorted(self._agents)]

    def with_capabilities(self, required: set[str] | frozenset[str]) -> list[AgentSpec]:
        return [a for a in self.all() if frozenset(required) <= a.capabilities]
