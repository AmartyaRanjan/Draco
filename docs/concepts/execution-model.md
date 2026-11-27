# Context, results and errors

Three objects carry everything about a run: the **context** going in, the **result** coming out, and the **errors** that explain a failure.

## The context

Your workload's first argument is an `ExecutionContext`. It is immutable and carries:

| Field | What it is |
|---|---|
| `execution_id` | A unique id for this run. The same id is used if the run is resumed. |
| `parent_execution_id`, `depth` | Where this run sits in a tree of nested runs. |
| `session_id` | A grouping id you choose, such as a chat session. Children inherit it. |
| `agent_id` | The name of the workload or agent. |
| `identity`, `user_id`, `tenant_id` | The trusted caller. When an identity is given, `user_id` and `tenant_id` are derived from it; without one they are the caller's own unauthenticated values. |
| `correlation_id` | One id shared by a run and all its descendants, for tying logs together. |
| `policy` | What this run may do: [tools, permissions, agents](security.md). |
| `budget` | The shared [counters and limits](budgets.md). |
| `cancel`, `deadline` | The cancellation token and the time limit. |
| `services` | The harness's service registry (events, store, tracer, authorizer, plugin services). |
| `metadata` | A dict for **your** per-request data. The context itself is immutable, but this dict is mutable. |

Two rules follow from this design:

- **Never keep request state in module-level variables.** Put it on `ctx.metadata`. Globals leak between concurrent requests, which in a multi-tenant system means between customers.
- **Call `ctx.check()` inside long loops.** It raises if the run was cancelled or ran past its deadline, so your code stops promptly instead of finishing work nobody wants.

### Nested runs

Pass `parent=ctx` to run another workload as a child. The child gets a new execution id and **shares everything that matters**: the budget, the identity, the correlation id and the session. Cancelling the parent cancels the child, but not the other way round.

<!-- doctest -->
```python
import asyncio
from draco import Harness, Identity

async def child(ctx, x):
    return {"user": ctx.user_id, "depth": ctx.depth, "tool_calls_so_far": ctx.budget.snapshot()["tool_calls"]}

async def parent(ctx, x):
    ctx.budget.charge_tool_call()                 # charged to the one shared budget
    c = await h.run("child", x, parent=ctx)
    return c.output

h = Harness()
h.register("child", child)
h.register("parent", parent)

async def main():
    r = await h.run("parent", 1, identity=Identity("alice", "acme"))
    print(r.output)
    print("children:", len(r.children), "| tool calls charged to the parent:", r.usage.tool_calls)

asyncio.run(main())
```

<!-- output -->
```text
{'user': 'alice', 'depth': 1, 'tool_calls_so_far': 1}
children: 1 | tool calls charged to the parent: 1
```

A child **cannot widen** its parent's policy, and it cannot change the user or tenant:

<!-- doctest -->
```python
import asyncio
from draco import Harness, Policy

async def child(ctx, x):
    return x

async def parent(ctx, x):
    wider = Policy(allowed_tools=frozenset({"delete_everything"}))     # the parent allows no tools
    c = await h.run("child", x, parent=ctx, policy=wider)
    return c.status.value, c.error.message

h = Harness()
h.register("child", child)
h.register("parent", parent)

async def main():
    r = await h.run("parent", 1, policy=Policy(allowed_tools=frozenset()))
    print(r.output)

asyncio.run(main())
```

<!-- output -->
```text
('failed', "a child execution cannot widen its parent's policy")
```

## The result

`Harness.run` and `Harness.resume` always return an `ExecutionResult`:

| Field | Meaning |
|---|---|
| `status` | `completed`, `failed`, `cancelled`, `timeout`, `budget_exhausted`, `handoff` or `waiting`. |
| `ok` | `True` only for `completed`. |
| `output` | What the workload returned (for `waiting`, the payload it waited with). |
| `error` | An `ErrorInfo` for a failure: `type`, `code`, a client-safe `message`, `retryable`, `details`. |
| `usage` | Model calls, tool calls, retries, tokens and duration charged to this run. |
| `handoff` | Set when the workload passed control to another agent. |
| `children` | The execution ids of nested runs. |
| `replayed` | `True` if this result came from an idempotency key instead of a fresh run. |
| `checkpoint_id` | Set for a `waiting` result: the id to resume. |

`result.raise_for_status()` turns a failed result back into the typed exception it started as, with the same `code`, `retryable` and `details`. The router uses this to propagate a child's failure.

## Errors

Every error Draco raises is a `DracoError` with a stable `code` and a `retryable` flag. The flag is set honestly, because it drives retries and HTTP status mapping: a timeout or a provider outage is retryable, a policy denial never is.

```text
DracoError
├── ConfigurationError   configuration     bad or missing setting
├── ContractError        contract          input or output failed validation
├── RoutingError         routing           unknown workload or agent, handoff loop
├── ToolError            tool              a tool failed
├── ProviderError        provider          a model or backend call failed
├── PolicyViolation      policy            denied by policy, authorization or tenant isolation
│   └── ApprovalRequired                   allowed only with an approval the policy lacks
├── ExecutionTimeout     timeout           deadline passed (retryable)
├── ExecutionCancelled   cancelled         cancelled by the caller or the parent
├── PersistenceError     persistence       the state store failed (retryable)
├── BudgetExhausted      budget            a limit was reached
└── PluginError          plugin            plugin registration or start failed
```

Anything that is **not** a `DracoError` is reduced to `unexpected error (TypeName)` in the result, with code `internal`. The message is hidden on purpose, so internal details, paths or secrets in an exception never reach a client. Raise a `DracoError` subclass when you want the caller to see a message.

The full table, with the HTTP status each maps to in the RAG service, is in [errors and statuses](../reference/errors.md).

## Next

[Security model](security.md).
