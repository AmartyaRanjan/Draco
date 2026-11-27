# State, steps and resume

Draco keeps three kinds of durable state for every run, in one **store**: the run's record, the events it emitted, and the checkpoints of its steps. This page explains what is kept, how `resume` uses it, and what Draco does and does not promise.

## What is stored

| Kind | Contents | Used for |
|---|---|---|
| **Execution record** | Id, target, status, timestamps, parent, session, agent, correlation id, idempotency key, attempts, input, output, error, usage, tenant, user, roles, the policy it ran under | Looking a run up, idempotency, resuming, retention |
| **Events** | The run's lifecycle and audit events, **redacted** | History and debugging |
| **Step checkpoints** | For each `step()`: `started` or `done`, and the result | Resuming without repeating finished work |

`h.store` gives you access: `await h.store.get_execution(id)`, `list_events(id)`, `list_steps(id)`, `list_children(id)`.

## Choosing a store

| Store | Survives a restart | Several processes | Use for |
|---|---|---|---|
| `InMemoryStore` (default) | No | No | Tests and short-lived scripts |
| `SqliteStore(path)` | Yes | One host | Single-machine services and development |
| `PostgresStore(dsn)` | Yes | Yes | Production with more than one process |

With no `store=` argument, the harness opens the one named in settings: `DRACO_STATE_URL` (a `postgresql://` URL) or `DRACO_STATE_PATH` (a SQLite file), otherwise memory. See [persist, encrypt and purge state](../guides/persistence.md).

If the store fails while a run is finishing, Draco logs it and keeps the result it computed, unless you create the harness with `strict_persistence=True`, in which case a completed run whose record cannot be saved is reported as failed.

## Steps

A **step** is a unit of work whose completion is recorded. In a workload:

```python
charged = await step(ctx, "charge", charge_card)
sent = await step(ctx, "email", send_receipt, idempotent=True)
```

`step(ctx, name, fn, idempotent=False)` behaves like this:

| What the store says about this step | What `step` does |
|---|---|
| Nothing yet | Runs `fn`, records `started`, then records `done` with the result |
| `done` | Returns the recorded result. **Does not run `fn`.** Emits `step.replayed`. |
| `started` (never finished), `idempotent=True` | Runs `fn` again |
| `started` (never finished), `idempotent=False` | Raises `ReconciliationRequired` |

The last row is the safety net. A step that started but never recorded completion has an **unknown** effect: the card may or may not have been charged. Draco will not guess for a step you have not promised is safe to repeat.

Step results must be JSON-serializable, because they are stored.

## Resume

`await h.resume(execution_id)` re-runs a run **under the same execution id**. It works for runs that ended `failed`, `timeout` or `cancelled`, runs that are `waiting`, and runs left `running` by a process that died. It refuses anything else (`completed`, `budget_exhausted`, `handoff`), and a run that is still executing in this process. A refusal raises `ContractError`, since `resume` is a call you make about an id, not a run.

On resume:

- The **input** is the stored input, unless you pass a new `input=`. This is how a human's answer reaches a `waiting` run.
- Completed steps are replayed from their records. Unfinished ones follow the table above.
- The run keeps its **policy**. You may pass a narrower one, but a policy that widens what the run started under is refused.
- The identity must belong to the run's tenant. A different tenant gets "unknown execution", the same answer as a missing id.
- `attempts` goes up by one, and the trace shows `execution.resumed`.

Resume re-runs your workload function from the top. That is why steps matter: code between steps runs again, so keep it free of side effects, and put every side effect inside a step.

## Waiting for a human

A workload can pause on purpose by returning `Waiting`:

```python
from draco import Waiting

async def refund(ctx, request):
    return Waiting(reason="needs approval", payload={"approve": request})
```

The run's status becomes `waiting`, its `output` is the payload, and the result carries a `checkpoint_id`. Later, `h.resume(execution_id, input=decision)` continues it. The [LangGraph guide](../guides/langgraph.md) shows the same pattern with a graph interrupt.

## What Draco promises, and what it does not

**Promises**

- A step that completed is not re-run on resume.
- A non-idempotent step with an unknown outcome is not blindly re-run: you get `ReconciliationRequired`.
- A completed run with the same idempotency key is returned, not re-executed.
- Resume is tenant-checked and cannot widen the policy.

**Does not promise**

- **Exactly-once effects.** If a step succeeded in the outside world but the process died before Draco recorded it, the step is `started`, not `done`. Make side effects safe to repeat, usually with an idempotency key sent to the other system, and mark the step `idempotent=True`.
- **Anything about code outside steps.** It runs again on resume.

## Retention and encryption

Old finished runs can be purged on a schedule, and `waiting` and `running` runs are never purged, because they can still be resumed. Payloads can be encrypted at rest. Both are covered in [persist, encrypt and purge state](../guides/persistence.md).

## Next

[Tracing and redaction](observability.md).
