"""Authorization decisions for protected operations. Fail closed: no identity, no matching grant,
or an exception inside the authorizer all deny."""
from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from draco.core.errors import PolicyViolation
from draco.core.identity import Identity


@dataclass(frozen=True)
class Resource:
    """What is being accessed. `tenant_id=None` means the resource is not tenant-scoped (e.g. a tool
    definition); a tenant-scoped resource is only reachable from the same tenant."""
    kind: str  # workload | agent | tool | document | retrieval | execution ...
    id: str = ""
    tenant_id: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str = ""

    @classmethod
    def allow(cls, reason: str = "") -> Decision:
        return cls(True, reason)

    @classmethod
    def deny(cls, reason: str) -> Decision:
        return cls(False, reason)


@runtime_checkable
class Authorizer(Protocol):
    def authorize(self, identity: Identity | None, action: str, resource: Resource) -> Awaitable[Decision] | Decision: ...


class RoleAuthorizer:
    """Default policy: tenant isolation first, then a role -> allowed-actions grant table.

    Actions are strings such as `workload.run`, `tool.call`, `document.read`; a grant may end in `*`
    (`tool.*`). With `grants=None` roles are not checked and only tenant isolation applies, which is
    what you want when per-execution `Policy` already carries the permission model.
    `require_identity=True` denies every protected operation that has no identity; so does configuring `grants`."""

    def __init__(self, grants: Mapping[str, set[str] | frozenset[str]] | None = None, *, require_identity: bool = False):
        self.grants = None if grants is None else {r: frozenset(a) for r, a in grants.items()}
        self.require_identity = require_identity

    def authorize(self, identity: Identity | None, action: str, resource: Resource) -> Decision:
        if identity is None:
            if self.require_identity or self.grants is not None:  # a grant table cannot be satisfied anonymously
                return Decision.deny("no identity established")
            if resource.tenant_id is not None:
                return Decision.deny("tenant-scoped resource requires an identity")
            return Decision.allow("no identity required")
        if resource.tenant_id is not None and resource.tenant_id != identity.tenant_id:
            return Decision.deny("cross-tenant access denied")
        if self.grants is None:
            return Decision.allow("tenant check passed")
        for role in identity.roles:
            for g in self.grants.get(role, ()):
                if g == action or (g.endswith("*") and action.startswith(g[:-1])):
                    return Decision.allow(f"role {role}")
        return Decision.deny(f"no role grants '{action}'")


async def decide(authorizer: Authorizer, identity: Identity | None, action: str, resource: Resource) -> Decision:
    """Evaluate an authorizer; an exception or a non-Decision result is a denial."""
    try:
        out = authorizer.authorize(identity, action, resource)
        if inspect.isawaitable(out):
            out = await out
    except Exception:
        return Decision.deny("authorizer failed")
    return out if isinstance(out, Decision) else Decision.deny("authorizer returned an invalid decision")


Check = Callable[[Identity | None, str, Resource], Awaitable[None]]


def raise_if_denied(decision: Decision, action: str, resource: Resource) -> None:
    if not decision.allowed:
        raise PolicyViolation(f"'{action}' on {resource.kind} '{resource.id}' denied: {decision.reason}",
                              details={"action": action, "resource_kind": resource.kind})
