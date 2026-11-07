"""Persistence contracts. Execution history, events and step checkpoints are kept apart from
conversational memory (which `app.inference.memory` already provides and Draco reuses)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ExecutionRecord:
    execution_id: str
    target: str
    status: str  # running | completed | failed | cancelled | timeout | budget_exhausted | handoff | waiting
    started_at: float
    parent_execution_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    correlation_id: str | None = None
    finished_at: float | None = None
    idempotency_key: str | None = None
    output: Any = None
    error: dict[str, Any] | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    attempts: int = 1
    input: Any = None  # as JSON; stored so a failed/waiting run can be resumed
    tenant_id: str | None = None  # owner; resume() and lookups are checked against the caller's tenant
    user_id: str | None = None
    roles: list[str] = field(default_factory=list)
    policy: dict[str, Any] | None = None  # the Policy the execution ran under; resume() cannot widen it


@dataclass
class StepRecord:
    execution_id: str
    step: str
    status: str  # started | done
    data: Any = None
    created_at: float = 0.0


@runtime_checkable
class ExecutionStore(Protocol):
    async def save_execution(self, rec: ExecutionRecord) -> None: ...
    async def get_execution(self, execution_id: str) -> ExecutionRecord | None: ...
    async def find_by_key(self, key: str) -> ExecutionRecord | None: ...
    async def list_children(self, execution_id: str) -> list[ExecutionRecord]: ...


@runtime_checkable
class EventStore(Protocol):
    async def append_event(self, execution_id: str, name: str, ts: float, data: dict[str, Any]) -> None: ...
    async def list_events(self, execution_id: str) -> list[dict[str, Any]]: ...


@runtime_checkable
class CheckpointStore(Protocol):
    async def begin_step(self, execution_id: str, step: str) -> None: ...
    async def complete_step(self, execution_id: str, step: str, data: Any) -> None: ...
    async def get_step(self, execution_id: str, step: str) -> StepRecord | None: ...
    async def list_steps(self, execution_id: str) -> list[StepRecord]: ...


ACTIVE_STATUSES = frozenset({"waiting", "running"})  # never purged: they can still be resumed
ENCRYPTED_PREFIX = "enc:v1:"  # marks an encrypted payload column; anything else is plaintext JSON


@dataclass(frozen=True)
class PurgeResult:
    executions: int = 0
    events: int = 0
    steps: int = 0

    def __add__(self, other: PurgeResult) -> PurgeResult:
        return PurgeResult(self.executions + other.executions, self.events + other.events, self.steps + other.steps)


@runtime_checkable
class RetentionStore(Protocol):
    async def purge(self, *, older_than: float | None = None,
                    events_older_than: float | None = None) -> PurgeResult:
        """Delete persisted state older than the given epoch-second cutoffs.

        `older_than`: terminal executions (any status except waiting/running) that finished before the
        cutoff, with their events and step checkpoints, plus orphaned events/steps older than it.
        `events_older_than`: events older than the cutoff; execution records are kept.
        Executions in `waiting` or `running` state, and every descendant of one, are never touched."""
        ...


@runtime_checkable
class PayloadCipher(Protocol):
    """Symmetric encryption for persisted payload columns (see `draco.security.crypto.FernetCipher`)."""

    def encrypt(self, plaintext: str) -> str: ...
    def decrypt(self, token: str) -> str: ...
    def rotate(self, token: str) -> str: ...  # re-encrypt under the current primary key


def dumps(v: Any) -> str:
    return json.dumps(v, default=str)
