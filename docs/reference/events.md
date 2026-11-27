# Events

Everything that happens in a run is an **event** on one bus. (Plugin events carry no execution id, so they are not stored per execution.) The tracer records them, the store saves them, exporters ship them, and your hooks can react to them.

An `Event` has a `name`, the ids it belongs to (`execution_id`, `parent_execution_id`, `correlation_id`), a `data` dict and a timestamp `ts`. The `data` you see in a trace or a store is already [redacted](../concepts/observability.md).

## Subscribe

`h.on(pattern, handler, critical=False)` subscribes to events. The pattern is an exact name, a prefix ending in `*` such as `tool.*`, or `*` for everything. It returns a function that unsubscribes. A handler may be sync or async.

- **Normal handlers** observe. If one raises, the error is recorded (`bus.handler_errors`) and the run continues.
- **Critical handlers** (`critical=True`) enforce. They run before normal handlers, and an exception from one (normally `PolicyViolation`) on `execution.started`, `tool.before` or `model.before` stops the run, the tool call or the model call.

<!-- doctest -->
```python
import asyncio
from draco import Harness

async def work(ctx, x):
    return x

async def main():
    h = Harness()
    h.register("work", work)

    seen = []
    h.on("execution.*", lambda event: seen.append((event.name, event.data.get("status"))))

    await h.run("work", 1)
    print(seen)

asyncio.run(main())
```

<!-- output -->
```text
[('execution.started', None), ('execution.finished', 'completed')]
```

Your own code can add events with `emit(ctx, name, **data)` from `draco.harness`; they are redacted like the rest.

## Execution

| Event | Emitted | Data |
|---|---|---|
| `execution.started` | A run is admitted and saved | `agent_id`, `target`, `depth`, `user_id`, `tenant_id`, `session_id` |
| `execution.resumed` | A resumed run begins (instead of `started`) | same as `started` |
| `execution.replayed` | An idempotency key returned a saved result | `key_reused` |
| `execution.failed`, `execution.cancelled`, `execution.timeout`, `execution.budget_exhausted` | A run ends with that status, just before `finished` | same as `finished` |
| `execution.finished` | Every run that was started or resumed, last. A replayed idempotent result, or a failure while building the context, emits no `finished`. | `status`, `agent_id`, `target`, `error_code`, `usage` |
| `handoff` | A workload returned a `Handoff` | `destination`, `reason` |
| `persistence.failed` | Saving the final record failed | `error` |

## Model calls and retries

| Event | Emitted | Data |
|---|---|---|
| `model.before` | Before a model call (a critical handler may veto) | `provider`, `model` |
| `model.after` | The call succeeded | `provider`, `model`, `tokens`, `latency_ms` |
| `model.failed` | The call raised | `provider`, `model`, `error`, `latency_ms` |
| `retry` | A retry is about to happen | from `call_with_retry`: `attempt`, `error`. From the tool runtime: `tool`, `attempt`, `error_code`. |

## Tools

| Event | Emitted | Data |
|---|---|---|
| `tool.before` | The call passed the policy and authorizer (a critical handler may veto) | `tool`, `purpose`, `group`, `mutates_state` |
| `tool.after` | A call that passed the checks finished, with a success, error or timeout result. Not emitted for refused calls (see `tool.denied`), or when the budget, the deadline or a cancellation ends the run. | `tool`, `status`, `group`, `retries`, `duration_ms`, `output_chars`, `error_code`, `untrusted`, `injection_suspected`, `args` (the first 20 argument **names** for dict arguments, never values) |
| `tool.denied` | The tool is unknown, or the policy or authorizer refused it | `tool`, `reason` |

## Steps and routing

| Event | Emitted | Data |
|---|---|---|
| `step.started` | A step begins | `step` |
| `step.completed` | A step finished and its result was saved | `step` |
| `step.replayed` | A resumed run returned a saved step result without running it | `step` |
| `step.reconciliation_required` | A non-idempotent step was found started but unfinished | `step` |
| `route.decided` | The router chose a destination | `destination`, `reason`, `strategy`, `hop` |

## Security and retrieval

| Event | Emitted | Data |
|---|---|---|
| `security.allowed`, `security.denied` | An authorizer decision | `action`, `resource_kind`, `resource_id`, `resource_tenant`, `reason`, `subject`. With the default settings `subject` and `resource_tenant` are pseudonymised, while `resource_id` (a tool name, a document id) is recorded as given. Never payloads. |
| `retrieval.after` | A retrieval workload finished | `selected`, `withheld`, `degraded` |

## Plugins

| Event | Emitted | Data |
|---|---|---|
| `plugin.started`, `plugin.stopped` | A plugin started or stopped | `plugin` |
| `plugin.failed` | A plugin failed during setup or start | `plugin`, `stage` |
