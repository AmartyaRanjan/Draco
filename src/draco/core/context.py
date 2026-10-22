"""Execution identity and control signals, propagated through nested work."""
from __future__ import annotations

import contextvars
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any

from draco.core.budget import BudgetTracker
from draco.core.errors import ExecutionCancelled, ExecutionTimeout, PolicyViolation
from draco.core.identity import Identity
from draco.core.policy import Policy


def new_id() -> str:
    return uuid.uuid4().hex


class CancellationToken:
    """Cooperative cancellation. Cancelling a parent cancels its children, not the reverse."""

    def __init__(self, parent: CancellationToken | None = None):
        self._cancelled = False
        self.reason: str | None = None
        self._callbacks: list[Callable[[], None]] = []
        self._children: list[CancellationToken] = []
        if parent is not None:
            parent._children.append(self)
            if parent._cancelled:
                self._cancelled, self.reason = True, parent.reason

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self, reason: str = "cancelled") -> None:
        if self._cancelled:
            return
        self._cancelled, self.reason = True, reason
        for cb in list(self._callbacks):
            cb()
        for child in list(self._children):
            child.cancel(reason)

    def on_cancel(self, cb: Callable[[], None]) -> None:
        self._callbacks.append(cb)
        if self._cancelled:
            cb()

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise ExecutionCancelled(self.reason or "cancelled")


@dataclass(frozen=True)
class ExecutionContext:
    execution_id: str = field(default_factory=new_id)
    parent_execution_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    user_id: str | None = None
    tenant_id: str | None = None
    identity: Identity | None = None  # trusted: set by the harness from verified credentials; inherited, never overridable
    correlation_id: str = field(default_factory=new_id)
    created_at: float = field(default_factory=time.time)
    cancel: CancellationToken = field(default_factory=CancellationToken)
    deadline: float | None = None  # time.monotonic() value
    budget: BudgetTracker = field(default_factory=BudgetTracker)
    policy: Policy = field(default_factory=Policy)
    depth: int = 0
    services: Any = None  # ServiceRegistry; typed Any to keep core free of plugin imports
    metadata: dict[str, Any] = field(default_factory=dict)  # per-execution, copied into children

    def child(self, *, agent_id: str | None = None, deadline: float | None = None,
              policy: Policy | None = None, metadata: dict[str, Any] | None = None) -> ExecutionContext:
        self.budget.check_depth(self.depth + 1)
        if policy is not None and not policy.narrows(self.policy):
            raise PolicyViolation("a child execution cannot widen its parent's policy")
        clash = RESERVED_METADATA & set(metadata or {})
        if clash:
            raise PolicyViolation(f"metadata cannot set trusted fields {sorted(clash)}")
        dls = [d for d in (self.deadline, deadline) if d is not None]
        return replace(self, execution_id=new_id(), parent_execution_id=self.execution_id,
                       agent_id=agent_id or self.agent_id, depth=self.depth + 1,
                       cancel=CancellationToken(self.cancel), deadline=min(dls) if dls else None,
                       policy=policy or self.policy, created_at=time.time(),
                       metadata={**self.metadata, **(metadata or {})})

    def time_left(self) -> float | None:
        return None if self.deadline is None else self.deadline - time.monotonic()

    def check(self) -> None:
        """Cooperative checkpoint: raises if cancelled, past deadline, or out of time budget."""
        self.cancel.raise_if_cancelled()
        left = self.time_left()
        if left is not None and left <= 0:
            raise ExecutionTimeout("execution deadline exceeded")
        self.budget.check_duration()

    def ids(self) -> dict[str, str | None]:
        return {"execution_id": self.execution_id, "parent_execution_id": self.parent_execution_id,
                "session_id": self.session_id, "agent_id": self.agent_id,
                "correlation_id": self.correlation_id}


RESERVED_METADATA = frozenset({"identity", "user_id", "tenant_id", "subject", "roles", "execution_id",
                               "parent_execution_id", "correlation_id", "policy", "authorizer"})

_current: contextvars.ContextVar[ExecutionContext | None] = contextvars.ContextVar("draco_ctx", default=None)


def current_context() -> ExecutionContext | None:
    return _current.get()


@contextmanager
def use_context(ctx: ExecutionContext) -> Iterator[ExecutionContext]:
    token = _current.set(ctx)
    try:
        yield ctx
    finally:
        _current.reset(token)
