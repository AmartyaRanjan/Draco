from __future__ import annotations

import copy
import time
from typing import Any

from draco.state.stores import ExecutionRecord, StepRecord


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
        return [copy.deepcopy(r) for r in self._exec.values() if r.parent_execution_id == execution_id]

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
