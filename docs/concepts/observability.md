# Tracing and redaction

Every step of a run emits an event on one bus. The **tracer** subscribes to that bus, removes secrets, keeps a trace per execution, and hands each record to any exporters you configure. Events are also saved to the store.

## What gets recorded

Lifecycle events (`execution.started`, `execution.finished`, ...), security events (`security.allowed`, `security.denied`), model and tool events, steps, handoffs and retries. The [event reference](../reference/events.md) lists them.

```python
h.tracer.trace(execution_id)     # the redacted events of one run, in order
h.tracer.tree(root_execution_id) # the run and its descendants as nested dicts
```

`tree` is the quickest way to see a multi-agent run. In the [routing guide](../guides/agents.md) it prints a call tree like `router → triage → math → auditor`.

## Redaction

The tracer cleans every event **before** it is stored, exported or returned:

- **Secret-looking keys.** Any key whose name contains `key`, `token`, `password`, `secret` or `authorization` has its value replaced with `<redacted>`.
- **Secret-looking strings.** `Bearer ...` tokens and `sk-...` keys inside text are replaced.
- **Personal identifiers.** `user_id`, `tenant_id`, `subject` and `resource_tenant` are replaced with a stable hash such as `id:2bd806c97f0e`. The same user always gets the same hash, so you can still count and correlate per user without storing the raw id.
- **Size.** Long strings are truncated (default 2000 characters) and large collections are capped.

<!-- doctest -->
```python
import asyncio
from draco import Harness, Identity
from draco.harness import emit

async def checkout(ctx, order):
    await emit(ctx, "app.checkout", api_key="sk-live-1234567890", amount=42,
               note="paid with Bearer abcdefgh12345678", user_id="alice")
    return order

async def main():
    h = Harness()
    h.register("checkout", checkout)
    r = await h.run("checkout", {"id": 7}, identity=Identity("alice", "acme"))
    event = next(e for e in h.tracer.trace(r.execution_id) if e["event"] == "app.checkout")
    print(event["data"])

asyncio.run(main())
```

<!-- output -->
```text
{'api_key': '<redacted>', 'amount': 42, 'note': 'paid with <redacted>', 'user_id': 'id:2bd806c97f0e'}
```

`emit(ctx, name, **data)` is how your own code adds events. They are redacted like all the others.

You can change which names are treated as secrets or personal identifiers with `DracoSettings.redact_keys` and `pii_keys` (`DRACO_REDACT_KEYS`, `DRACO_PII_KEYS`), and the size limit with `max_payload_chars`.

!!! warning "Redaction covers events, not the execution record"
    The execution record stores your run's **input and output as given**, because `resume` needs them. If a workload receives `{"password": "..."}` as its input, that value is in the record. Keep secrets out of workload inputs and outputs (pass a reference, not the secret), and [encrypt state at rest](../guides/persistence.md) if the store holds anything sensitive.

Tool events record the argument **names**, never the values.

## Exporters

An exporter is any object with an `export(record: dict)` method. Draco ships:

| Exporter | Does |
|---|---|
| `MemoryExporter` | Keeps records in a list (the default when nothing else is configured) |
| `JsonlExporter(path)` | Appends one JSON line per event to a file. Also enabled by `DRACO_TRACE_PATH`. |
| `OtelExporter(tracer_provider)` | One OpenTelemetry span per execution, parented by the execution tree (needs `draco-harness[otel]`) |

An exporter that raises cannot break a run: the bus records the error and continues. See [export traces](../guides/tracing.md).

## Limits

The in-memory trace is **per process** and has no retention option. For history that survives a restart, use the persisted events in the store, or export to a file or to OpenTelemetry.

## Next

You have the model. Put it to work with the [how-to guides](../guides/tools.md), or look things up in the [reference](../reference/configuration.md).
