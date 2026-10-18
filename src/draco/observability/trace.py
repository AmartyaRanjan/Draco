"""Event-driven tracing: every bus event becomes a redacted record, exported to pluggable sinks."""
from __future__ import annotations

import json
import threading
from typing import Any, Protocol

from draco.core.events import Event, EventBus
from draco.observability.redaction import Redactor


class Exporter(Protocol):
    def export(self, record: dict[str, Any]) -> None: ...


class MemoryExporter:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def export(self, record: dict[str, Any]) -> None:
        self.records.append(record)


class JsonlExporter:
    """Append-only local trace file; no hosted platform required."""

    def __init__(self, path: str):
        self.path, self._lock = path, threading.Lock()

    def export(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=str)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class Tracer:
    """Subscribes to the bus. Keeps an in-memory trace per execution and fans out to exporters.
    Exporter failures are swallowed by the bus (non-critical handler) so tracing never breaks a run."""

    def __init__(self, bus: EventBus, exporters: list[Exporter] | None = None, redactor: Redactor | None = None):
        self.redactor = redactor or Redactor()
        self.exporters: list[Exporter] = exporters if exporters is not None else [MemoryExporter()]
        self._by_exec: dict[str, list[dict[str, Any]]] = {}
        bus.subscribe("*", self._on_event)

    def _on_event(self, e: Event) -> None:
        rec = {"event": e.name, "ts": e.ts, "execution_id": e.execution_id,
               "parent_execution_id": e.parent_execution_id, "correlation_id": e.correlation_id,
               "data": self.redactor.redact(e.data)}
        if e.execution_id:
            self._by_exec.setdefault(e.execution_id, []).append(rec)
        for ex in self.exporters:
            ex.export(rec)

    def trace(self, execution_id: str) -> list[dict[str, Any]]:
        return list(self._by_exec.get(execution_id, []))

    def tree(self, root_execution_id: str) -> dict[str, Any]:
        """Parent/child execution tree reconstructed from recorded events."""
        children: dict[str, list[str]] = {}
        for eid, recs in self._by_exec.items():
            parent = recs[0]["parent_execution_id"]
            if parent:
                children.setdefault(parent, []).append(eid)

        def build(eid: str) -> dict[str, Any]:
            recs = self._by_exec.get(eid, [])
            agent = next((r["data"].get("agent_id") for r in recs if r["event"] == "execution.started"), None)
            status = next((r["data"].get("status") for r in reversed(recs) if r["event"] == "execution.finished"), None)
            return {"execution_id": eid, "agent_id": agent, "status": status,
                    "children": [build(c) for c in children.get(eid, [])]}
        return build(root_execution_id)
