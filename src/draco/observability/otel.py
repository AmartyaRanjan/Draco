"""Optional OpenTelemetry exporter (`pip install draco[otel]`): one span per execution, parented by
the Draco parent/child relationship, with lifecycle/tool/model events attached as span events.
It is an `Exporter` for the existing Tracer, not a second tracing system."""
from __future__ import annotations

from typing import Any

_SPAN_END = {"execution.finished"}


class OtelExporter:
    def __init__(self, tracer_provider: Any = None, name: str = "draco"):
        from opentelemetry import trace
        self._trace = trace
        self._tracer = (tracer_provider or trace.get_tracer_provider()).get_tracer(name)
        self._spans: dict[str, Any] = {}

    def export(self, record: dict[str, Any]) -> None:
        eid, ev, data = record.get("execution_id"), record["event"], record["data"]
        if not eid:
            return
        ns = int(record["ts"] * 1e9)
        if ev in ("execution.started", "execution.resumed") and eid not in self._spans:
            parent = self._spans.get(record.get("parent_execution_id") or "")
            ctx = self._trace.set_span_in_context(parent) if parent is not None else None
            span = self._tracer.start_span(f"execution {data.get('target') or data.get('agent_id')}", context=ctx,
                                           start_time=ns, attributes={
                                               "draco.execution_id": eid,
                                               "draco.parent_execution_id": record.get("parent_execution_id") or "",
                                               "draco.correlation_id": record.get("correlation_id") or "",
                                               "draco.agent_id": str(data.get("agent_id") or ""),
                                               "draco.depth": int(data.get("depth") or 0)})
            self._spans[eid] = span
            return
        span = self._spans.get(eid)
        if span is None:
            return
        flat = {k: v for k, v in data.items() if isinstance(v, (str, int, float, bool))}
        span.add_event(ev, attributes=flat, timestamp=ns)
        if ev in _SPAN_END:
            status = str(data.get("status", ""))
            span.set_attribute("draco.status", status)
            if status not in ("completed", "handoff", "waiting"):
                from opentelemetry.trace import Status, StatusCode
                span.set_status(Status(StatusCode.ERROR, str(data.get("error_code") or status)))
            span.end(end_time=ns)
            del self._spans[eid]
