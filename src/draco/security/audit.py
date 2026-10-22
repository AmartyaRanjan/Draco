"""Audit records for security decisions, carried on the same event bus as the rest of the lifecycle
(so they get the same redaction, persistence and execution identifiers)."""
from __future__ import annotations

from typing import Any

from draco.core.context import ExecutionContext
from draco.harness.instrument import emit
from draco.security.authorization import Authorizer, Decision, Resource, decide, raise_if_denied


async def audit(ctx: ExecutionContext, action: str, resource: Resource, decision: Decision, **extra: Any) -> None:
    """Emits `security.allowed` / `security.denied`. Records ids and reasons, never payloads."""
    await emit(ctx, "security.allowed" if decision.allowed else "security.denied", action=action,
               resource_kind=resource.kind, resource_id=resource.id, resource_tenant=resource.tenant_id,
               reason=decision.reason, subject=ctx.identity.subject if ctx.identity else None, **extra)


async def authorize(ctx: ExecutionContext, action: str, resource: Resource, *, authorizer: Authorizer | None = None) -> None:
    """Authorize `ctx`'s identity for `action`, audit the outcome, raise PolicyViolation on denial.
    Uses the `authorizer` service of the harness unless one is given. If none is configured at all the
    operation is denied, not allowed."""
    if authorizer is None:
        s = ctx.services
        authorizer = s.get("authorizer") if s is not None and s.has("authorizer") else None
    decision = Decision.deny("no authorizer configured") if authorizer is None else await decide(
        authorizer, ctx.identity, action, resource)
    await audit(ctx, action, resource, decision)
    raise_if_denied(decision, action, resource)
