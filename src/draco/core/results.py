"""Normalized outcome of any execution."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from draco.core.errors import BudgetExhausted, DracoError, ExecutionCancelled, ExecutionTimeout


class Status(StrEnum):
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"
    timeout = "timeout"
    budget_exhausted = "budget_exhausted"
    handoff = "handoff"
    waiting = "waiting"


@dataclass
class ErrorInfo:
    type: str
    code: str
    message: str  # client-safe: unknown exceptions are reduced to their type name
    retryable: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_exception(cls, exc: BaseException) -> ErrorInfo:
        if isinstance(exc, DracoError):
            return cls(type(exc).__name__, exc.code, exc.message, exc.retryable, dict(exc.details))
        return cls(type(exc).__name__, "internal", f"unexpected error ({type(exc).__name__})")


@dataclass
class Usage:
    model_calls: int = 0
    tool_calls: int = 0
    retries: int = 0
    tokens: int = 0
    duration_ms: float = 0.0


@dataclass
class Handoff:
    destination: str
    reason: str = ""
    input: Any = None


@dataclass
class Waiting:
    """Return from a workload to pause: the execution becomes resumable via Harness.resume()."""
    reason: str = ""
    payload: Any = None


@dataclass
class ExecutionResult:
    status: Status
    execution_id: str
    output: Any = None
    error: ErrorInfo | None = None
    parent_execution_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    correlation_id: str | None = None
    usage: Usage = field(default_factory=Usage)
    handoff: Handoff | None = None
    checkpoint_id: str | None = None  # set for waiting/resumable results
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    children: list[str] = field(default_factory=list)  # child execution ids
    replayed: bool = False  # True when returned from an idempotency key instead of re-executing

    @property
    def ok(self) -> bool:
        return self.status == Status.completed

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    def raise_for_status(self) -> None:
        if self.ok:
            return
        e = self.error
        if self.status == Status.timeout:
            raise ExecutionTimeout(e.message if e else "timeout")
        if self.status == Status.cancelled:
            raise ExecutionCancelled(e.message if e else "cancelled")
        if self.status == Status.budget_exhausted:
            raise BudgetExhausted((e.details.get("resource") if e else None) or "unknown",
                                  (e.details.get("limit") if e else 0) or 0)
        raise DracoError(e.message if e else f"execution ended with status {self.status.value}")


def status_for(exc: BaseException) -> Status:
    if isinstance(exc, ExecutionTimeout):
        return Status.timeout
    if isinstance(exc, ExecutionCancelled):
        return Status.cancelled
    if isinstance(exc, BudgetExhausted):
        return Status.budget_exhausted
    return Status.failed
