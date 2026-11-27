# Export traces

By default a harness keeps each run's [redacted](../concepts/observability.md) trace in memory, which is enough for tests and a quick look. To send traces somewhere durable, give the harness one or more **exporters**. Every event goes to every exporter, already redacted.

An exporter that raises can never break a run: the failure is recorded and the run continues.

!!! note "Passing `exporters=` replaces the default"
    `Harness(exporters=[...])` uses exactly the list you give. `h.tracer.trace(execution_id)` keeps working, because the tracer holds each run's events itself, but nothing else is exported unless you list it.

## To a file

`JsonlExporter(path)` appends one JSON object per event. You can also switch it on without code by setting `DRACO_TRACE_PATH`.

Each record has `event`, `ts`, `execution_id`, `parent_execution_id`, `correlation_id` and a redacted `data` dict. That is enough to rebuild any run, and to group a whole request by `correlation_id`.

## To your own system

An exporter is any object with an `export(record: dict)` method. Keep it fast and do not raise: it runs on the event path. This one counts events, and a second exporter writes the file:

<!-- doctest -->
```python
import asyncio
import json
import os
import tempfile

from draco import Harness, Identity
from draco.observability import JsonlExporter

class EventCounter:
    def __init__(self):
        self.counts = {}

    def export(self, record):
        self.counts[record["event"]] = self.counts.get(record["event"], 0) + 1

async def work(ctx, x):
    return x

async def main():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "trace.jsonl")
        counter = EventCounter()
        h = Harness(exporters=[JsonlExporter(path), counter])
        h.register("work", work)
        await h.run("work", 1, identity=Identity("alice", "acme"))

        records = [json.loads(line) for line in open(path)]
        print([r["event"] for r in records])
        print(sorted(records[0]))
        print("user id in the file:", records[0]["data"]["user_id"])
        print(counter.counts)

asyncio.run(main())
```

<!-- output -->
```text
['execution.started', 'security.allowed', 'execution.finished']
['correlation_id', 'data', 'event', 'execution_id', 'parent_execution_id', 'ts']
user id in the file: id:2bd806c97f0e
{'execution.started': 1, 'security.allowed': 1, 'execution.finished': 1}
```

The user id in the file is the pseudonym, not `alice`. Redaction happens before any exporter sees the event.

## To OpenTelemetry

`OtelExporter` turns each execution into **one span**, parented by Draco's parent and child relationship, and attaches the lifecycle, tool and model events to it as span events. It is an exporter for the existing tracer, not a second tracing system, so what you see in OpenTelemetry is exactly the redacted trace.

```bash
pip install "draco-harness[otel]"
```

The example uses an in-memory span exporter so it needs no collector. In production you would build your `TracerProvider` with an OTLP exporter instead:

<!-- doctest requires=opentelemetry -->
```python
import asyncio

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from draco import Harness
from draco.observability.otel import OtelExporter

memory = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(memory))

h = Harness(exporters=[OtelExporter(provider)])

async def child(ctx, x):
    return x

async def parent(ctx, x):
    return (await h.run("child", x, parent=ctx)).output

h.register("parent", parent)
h.register("child", child)

async def main():
    await h.run("parent", 1)
    for span in sorted(memory.get_finished_spans(), key=lambda s: s.start_time):
        print(span.name, "| has parent span:", span.parent is not None, "| status:", span.attributes["draco.status"])

asyncio.run(main())
```

<!-- output -->
```text
execution parent | has parent span: False | status: completed
execution child | has parent span: True | status: completed
```

A run that does not end `completed`, `handoff` or `waiting` marks its span as an error, with the error code as the description.

## Persisted events

Separately from exporters, the harness saves each event, redacted, to the **store** (`h.store.list_events(execution_id)`). They live as long as your [retention settings](persistence.md) allow, and survive a restart if the store does. Use them for history; use exporters for live monitoring.

## See also

- [Tracing and redaction](../concepts/observability.md): what is removed before export
- [Events reference](../reference/events.md)
