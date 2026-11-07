from __future__ import annotations

import copy
import time
from typing import Any

from draco.state.stores import ACTIVE_STATUSES, ExecutionRecord, PurgeResult, StepRecord


class InMemoryStore:
    """Development/test backend implementing ExecutionStore, EventStore and CheckpointStore."""

    def __init__(self) -> None:
        self._exec: dict[str, ExecutionRecord] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._steps: dict[tuple[str, str], StepRecord] = {}

    async def save_execution(self, rec: ExecutionRecord) -> None:
        self._exec[rec.execution_id] = copy.deepcopy(rec)

    async def get_execution(self, execution_id: str) -> ExecutionRecord | None:
        r = self._exec.get(execution_id)
        return copy.deepcopy(r) if r else None

    async def find_by_key(self, key: str) -> ExecutionRecord | None:
        for r in self._exec.values():
            if r.idempotency_key == key:
                return copy.deepcopy(r)
        return None

    async def list_children(self, execution_id: str) -> list[ExecutionRecord]:
        return sorted((copy.deepcopy(r) for r in self._exec.values() if r.parent_execution_id == execution_id),
                      key=lambda r: r.started_at)  # same order as the SQL stores

    async def append_event(self, execution_id: str, name: str, ts: float, data: dict[str, Any]) -> None:
        self._events.setdefault(execution_id, []).append({"name": name, "ts": ts, "data": copy.deepcopy(data)})

    async def list_events(self, execution_id: str) -> list[dict[str, Any]]:
        return copy.deepcopy(self._events.get(execution_id, []))

    async def begin_step(self, execution_id: str, step: str) -> None:
        self._steps[(execution_id, step)] = StepRecord(execution_id, step, "started", None, time.time())

    async def complete_step(self, execution_id: str, step: str, data: Any) -> None:
        self._steps[(execution_id, step)] = StepRecord(execution_id, step, "done", copy.deepcopy(data), time.time())

    async def get_step(self, execution_id: str, step: str) -> StepRecord | None:
        s = self._steps.get((execution_id, step))
        return copy.deepcopy(s) if s else None

    async def list_steps(self, execution_id: str) -> list[StepRecord]:
        return sorted((copy.deepcopy(s) for (e, _), s in self._steps.items() if e == execution_id),
                      key=lambda s: s.created_at)

    def _protected(self) -> set[str]:
        """Waiting/running executions and all of their descendants."""
        keep = {i for i, r in self._exec.items() if r.status in ACTIVE_STATUSES}
        grew = True
        while grew:
            more = {i for i, r in self._exec.items() if r.parent_execution_id in keep} - keep
            keep |= more
            grew = bool(more)
        return keep

    async def purge(self, *, older_than: float | None = None, events_older_than: float | None = None) -> PurgeResult:
        keep = self._protected()
        n_exec = n_events = n_steps = 0
        if older_than is not None:
            victims = {i for i, r in self._exec.items() if i not in keep and r.status not in ACTIVE_STATUSES
                       and (r.finished_at if r.finished_at is not None else r.started_at) < older_than}
            for i in victims:
                del self._exec[i]
                n_events += len(self._events.pop(i, []))
            for k in [k for k, s in self._steps.items()
                      if k[0] in victims or (k[0] not in self._exec and s.created_at < older_than)]:
                del self._steps[k]
                n_steps += 1
            n_exec = len(victims)
            for i in [i for i in self._events if i not in self._exec]:  # orphaned events
                n_events += self._drop_events(i, older_than)
        if events_older_than is not None:
            for i in [i for i in self._events if i not in keep]:
                n_events += self._drop_events(i, events_older_than)
        return PurgeResult(n_exec, n_events, n_steps)

    def _drop_events(self, execution_id: str, before: float) -> int:
        evs = self._events[execution_id]
        kept = [e for e in evs if e["ts"] >= before]
        if kept:
            self._events[execution_id] = kept
        else:
            del self._events[execution_id]
        return len(evs) - len(kept)
