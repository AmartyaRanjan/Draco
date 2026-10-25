"""Authorization data for one execution. Enforced by the runtime, never by prompts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from draco.core.errors import PolicyViolation


@dataclass(frozen=True)
class Policy:
    permissions: frozenset[str] = frozenset({"read"})
    allowed_tools: frozenset[str] | None = None  # None = no tools are permitted
    denied_tools: frozenset[str] = field(default_factory=frozenset)
    allowed_agents: frozenset[str] | None = None
    allow_approval_required: bool = False

    def authorize_tool(self, name: str, required: frozenset[str] = frozenset(), *,
                       requires_approval: bool = False) -> None:
        if name in self.denied_tools:
            raise PolicyViolation(f"tool '{name}' is denied by policy")
        if self.allowed_tools is None or name not in self.allowed_tools:
            raise PolicyViolation(f"tool '{name}' is not allowed for this execution")
        missing = required - self.permissions
        if missing:
            raise PolicyViolation(f"tool '{name}' needs permissions {sorted(missing)}")
        if requires_approval and not self.allow_approval_required:
            raise PolicyViolation(f"tool '{name}' requires approval")

    def authorize_agent(self, agent_id: str) -> None:
        if self.allowed_agents is not None and agent_id not in self.allowed_agents:
            raise PolicyViolation(f"agent '{agent_id}' is not allowed for this execution")

    def narrows(self, parent: Policy) -> bool:
        """True if this policy grants nothing the parent does not. Child executions may only narrow."""
        def within(child: frozenset[str] | None, par: frozenset[str] | None) -> bool:  # None = unrestricted
            return par is None or (child is not None and child <= par)
        tools_ok = (self.allowed_tools or frozenset()) <= (parent.allowed_tools or frozenset())  # None = no tools
        return (self.permissions <= parent.permissions and tools_ok
                and within(self.allowed_agents, parent.allowed_agents)
                and parent.denied_tools <= self.denied_tools
                and (parent.allow_approval_required or not self.allow_approval_required))

    def to_dict(self) -> dict[str, Any]:
        def lst(v: frozenset[str] | None) -> list[str] | None:
            return None if v is None else sorted(v)
        return {"permissions": sorted(self.permissions), "allowed_tools": lst(self.allowed_tools),
                "denied_tools": sorted(self.denied_tools), "allowed_agents": lst(self.allowed_agents),
                "allow_approval_required": self.allow_approval_required}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Policy:
        def fs(v: list[str] | None) -> frozenset[str] | None:
            return None if v is None else frozenset(v)
        return cls(permissions=frozenset(d.get("permissions", ())), allowed_tools=fs(d.get("allowed_tools")),
                   denied_tools=frozenset(d.get("denied_tools", ())), allowed_agents=fs(d.get("allowed_agents")),
                   allow_approval_required=bool(d.get("allow_approval_required", False)))
