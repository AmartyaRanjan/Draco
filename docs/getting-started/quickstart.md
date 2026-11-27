# Quickstart

In about five minutes you will run a workload, read its result, see what happens when it fails, and look at its trace. Everything here needs only `pip install draco-harness`.

## 1. A workload

A **workload** is any `async def` that takes a context and an input. Register it under a name and run it by name:

<!-- doctest -->
```python
import asyncio
from draco import Harness

async def greeter(ctx, name):
    return f"hello {name}"

async def main():
    h = Harness()
    h.register("greeter", greeter)
    result = await h.run("greeter", "world")
    print(result.status, result.ok, result.output)

asyncio.run(main())
```

<!-- output -->
```text
completed True hello world
```

`ctx` is an [`ExecutionContext`](../reference/core.md): the ids of this run, the identity of the caller, the policy, the budget and the deadline. A greeter doesn't need it, but anything that calls a model, runs a tool or starts nested work does.

## 2. Failures come back as results

`Harness.run` does not raise when your workload fails. It returns an [`ExecutionResult`](../reference/core.md) with a `status` and, on failure, an `error`:

<!-- doctest -->
```python
import asyncio
from draco import DracoError, Harness
from draco.core import PolicyViolation

async def buggy(ctx, _):
    raise ValueError("database password is hunter2")

async def forbidden(ctx, _):
    raise PolicyViolation("not for you")

async def main():
    h = Harness()
    h.register("buggy", buggy)
    h.register("forbidden", forbidden)

    r = await h.run("buggy")
    print(r.status, r.error.code, r.error.message, r.error.retryable)

    r = await h.run("forbidden")
    print(r.status, r.error.code, r.error.message, r.error.retryable)

    try:
        r.raise_for_status()          # turn the result back into an exception when you want one
    except DracoError as exc:
        print(type(exc).__name__, exc.code)

asyncio.run(main())
```

<!-- output -->
```text
failed internal unexpected error (ValueError) False
failed policy not for you False
PolicyViolation policy
```

Two things to notice:

- The first error's message is `unexpected error (ValueError)`, **not** the text of the exception. Draco only passes on messages from its own typed errors, so a stray `"database password is hunter2"` never reaches a client. The original exception is logged by type only.
- A `PolicyViolation` keeps its message and its type, and `raise_for_status()` re-raises the same type, so a caller can still tell a denial from an outage.

The statuses are `completed`, `failed`, `cancelled`, `timeout`, `budget_exhausted`, `handoff` and `waiting`. See [errors and statuses](../reference/errors.md).

## 3. Time limits

Pass `timeout_s` to bound a run. When it passes, the workload is cancelled and the result says so:

<!-- doctest -->
```python
import asyncio
from draco import Harness

async def slow(ctx, _):
    await asyncio.sleep(5)

async def main():
    h = Harness()
    h.register("slow", slow)
    r = await h.run("slow", timeout_s=0.05)
    print(r.status, r.error.code, r.error.retryable)

asyncio.run(main())
```

<!-- output -->
```text
timeout timeout True
```

Without `timeout_s`, the default is 120 seconds (`DracoSettings.default_timeout_s`).

## 4. Who is calling

Pass an `Identity` to say who is acting. It carries a subject, a **mandatory tenant**, and optional roles. Draco derives `user_id` and `tenant_id` from it and records them with the run:

<!-- doctest -->
```python
import asyncio
from draco import Harness, Identity

async def whoami(ctx, _):
    return {"user": ctx.user_id, "tenant": ctx.tenant_id}

async def main():
    h = Harness()
    h.register("whoami", whoami)
    r = await h.run("whoami", identity=Identity("alice", "acme"), session_id="chat-1")
    print(r.output)
    saved = await h.store.get_execution(r.execution_id)
    print(saved.status, saved.user_id, saved.tenant_id, saved.session_id)

asyncio.run(main())
```

<!-- output -->
```text
{'user': 'alice', 'tenant': 'acme'}
completed alice acme chat-1
```

The identity is what Draco's security checks use. The [multi-tenant tutorial](../tutorials/multi-tenant.md) shows what that buys you.

## 5. The trace

Every run emits events, which the harness records. Ask for the trace of one execution:

<!-- doctest -->
```python
import asyncio
from draco import Harness

async def work(ctx, x):
    return x

async def main():
    h = Harness()
    h.register("work", work)
    r = await h.run("work", 1)
    print([e["event"] for e in h.tracer.trace(r.execution_id)])

asyncio.run(main())
```

<!-- output -->
```text
['execution.started', 'security.allowed', 'execution.finished']
```

`security.allowed` is the audit event for the `workload.run` authorization check. See [tracing and redaction](../concepts/observability.md) for what is recorded and what is removed.

## Where next

- **Learn by building:** [Guard a tool-using agent](../tutorials/guarded-tools.md) is the best next stop. It adds tools, policy and budgets.
- **Understand the model:** [How a run works](../concepts/lifecycle.md) explains every step between `run()` and the result.
