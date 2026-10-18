import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

from draco.core import DracoSettings  # noqa: E402
from draco.harness import Harness  # noqa: E402
from draco.observability.otel import OtelExporter  # noqa: E402
from draco.observability.trace import MemoryExporter  # noqa: E402


async def test_nested_executions_become_parented_spans_with_error_status():
    mem = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(mem))
    h = Harness(DracoSettings(), exporters=[MemoryExporter(), OtelExporter(provider)])

    async def child(ctx, x):
        raise ValueError("nope")

    async def parent(ctx, x):
        await h.run("child", x, parent=ctx)
        return "done"
    h.register("child", child)
    h.register("parent", parent)
    r = await h.run("parent", 1)
    spans = {s.attributes["draco.agent_id"]: s for s in mem.get_finished_spans()}
    assert set(spans) == {"parent", "child"}
    assert spans["child"].parent.span_id == spans["parent"].context.span_id
    assert spans["child"].status.status_code.name == "ERROR" and spans["parent"].status.status_code.name == "UNSET"
    assert spans["parent"].attributes["draco.execution_id"] == r.execution_id
