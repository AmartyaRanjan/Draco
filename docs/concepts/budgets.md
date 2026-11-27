# Budgets and limits

An agent loop is only as safe as its stopping condition. Draco counts what a run uses and ends it when a ceiling is hit, in code, regardless of what the model wants to do next.

## What is counted

A `BudgetLimits` sets the ceilings. Every field is optional (`None` is unlimited) except `max_depth`, which defaults to 8.

| Limit | Counts | When it is charged |
|---|---|---|
| `max_model_calls` | Calls to a model | Each `call_model` / `generate` |
| `max_tokens` | Tokens used | When the provider reports usage |
| `max_tool_calls` | Tool calls | Each `ToolRuntime.call` that gets past the policy checks |
| `max_retries` | Retries | Each retry made by `call_with_retry` |
| `max_depth` | Nesting of child runs | Each `h.run(..., parent=ctx)` |
| `max_duration_s` | Wall-clock seconds | Bounds the run's deadline. Exceeding it ends the run as `timeout`. |

All values must be positive. Pass limits per run (`h.run(..., limits=BudgetLimits(...))`) or set defaults for every run on `DracoSettings` (`max_model_calls`, `max_tool_calls`, `max_tokens`, ... or `DRACO_MAX_MODEL_CALLS` and friends).

## One budget, shared down the tree

A run and all of its children draw from **one** `BudgetTracker`. If a parent spawns ten children, they do not each get a fresh allowance. This is what stops an agent from escaping its limit by delegating.

## What happens at the limit

Hitting a limit ends the **whole run** with status `budget_exhausted`. The error's `details` name the resource and the limit. (The one exception is time: a run that exceeds `max_duration_s` or its deadline ends as `timeout`, see below.)



<!-- doctest -->
```python
import asyncio
from draco import BudgetLimits, Harness, call_model

async def agent(ctx, n):
    async def fake_model():
        return "text"
    for _ in range(n):
        await call_model(ctx, fake_model, provider="fake", model="demo")
    return "done"

async def main():
    h = Harness()
    h.register("agent", agent)
    limits = BudgetLimits(max_model_calls=2)

    r = await h.run("agent", 3, limits=limits)
    print(r.status, r.error.code, r.error.details, "| model calls charged:", r.usage.model_calls)

    r = await h.run("agent", 2, limits=limits)
    print(r.status, "| model calls charged:", r.usage.model_calls)

asyncio.run(main())
```

<!-- output -->
```text
budget_exhausted budget {'resource': 'model_calls', 'limit': 2} | model calls charged: 2
completed | model calls charged: 2
```

The third call is not allowed to start; the run is stopped at the limit.

`call_model(ctx, fn, ..., provider=, model=, usage_of=)` is the wrapper that charges the budget and emits `model.before` / `model.after` events for any model call. The [model provider guide](../guides/model-providers.md) shows it with real providers.

## Deadlines and timeouts

Every run has a deadline: `timeout_s` if you pass one, otherwise `DracoSettings.default_timeout_s` (120 s). When it passes, the workload task is cancelled and the result is `timeout` with `retryable=True`. A child's deadline can only be earlier than its parent's.

Cancellation works the same way from the outside. Give the run a `CancellationToken` and cancel it when you need to; the workload task is cancelled and the result says why. Inside a long loop, call `ctx.check()` so the run notices promptly:

<!-- doctest -->
```python
import asyncio
from draco import Harness
from draco.core import CancellationToken

async def loop_forever(ctx, _):
    while True:
        ctx.check()                     # raises if cancelled or past the deadline
        await asyncio.sleep(0.01)

async def main():
    h = Harness()
    h.register("loop", loop_forever)
    token = CancellationToken()
    task = asyncio.create_task(h.run("loop", cancel=token))
    await asyncio.sleep(0.05)
    token.cancel("user pressed stop")
    r = await task
    print(r.status, "|", r.error.message)

asyncio.run(main())
```

<!-- output -->
```text
cancelled | user pressed stop
```

If you know the execution id instead (for example from the `execution.started` event), `h.cancel(execution_id, reason)` does the same. A tenant-scoped run can only be cancelled with an `identity=` from the same tenant; otherwise `cancel` returns `False`, as if the run did not exist.

## Limits on who may run

Budgets bound one run. Two more limits bound how many runs start:

- **Concurrency.** `DracoSettings.max_concurrency` bounds top-level runs executing at once in this process; extra runs **wait** for a free slot, and the wait counts against their deadline. `max_concurrency_per_tenant` bounds one tenant's runs in flight; beyond it, runs are **rejected** with `budget_exhausted` (resource `tenant_concurrency`), not queued.
- **Rate limits.** Per-tenant and per-user token buckets. See [rate-limit tenants and users](../guides/rate-limits.md).

## Next

[State, steps and resume](state.md).
