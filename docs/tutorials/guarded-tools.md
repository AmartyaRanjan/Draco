# Tutorial: guard a tool-using agent

**You will build** an agent that can call tools, and see Draco refuse the calls that should not happen: a tool nobody allowed, a tool that needs a permission the run lacks, arguments that fail validation, and a runaway loop of calls.

**You need** `pip install draco-harness` and about ten minutes. No model or API key is involved; a small function stands in for the model, because the interesting part is what happens *around* the model's tool requests.

## The idea

A model decides which tool to call and with what arguments. Both are untrusted input. If the only thing between a model's request and your database is a sentence in the prompt, a clever user can talk their way past it. Draco puts the checks in code, in the tool runtime, and runs them **before your tool function executes**.

## 1. Define two tools

A tool is an async function plus a description of its input. The input is a `pydantic` model, so the arguments are validated before your function sees them:

```python
from pydantic import BaseModel, Field
from draco.tools import ToolRegistry, native_tool

registry = ToolRegistry()

class Convert(BaseModel):
    celsius: float = Field(ge=-273.15)          # colder than absolute zero is invalid

@native_tool(registry, name="to_fahrenheit", description="Convert Celsius to Fahrenheit",
             input_model=Convert)
async def to_fahrenheit(args, ctx):
    return {"fahrenheit": args.celsius * 9 / 5 + 32}

class Nothing(BaseModel):
    pass

@native_tool(registry, name="delete_records", description="Delete all records",
             input_model=Nothing, permissions={"write"}, mutates_state=True)
async def delete_records(args, ctx):
    return {"deleted": 3}
```

The second tool is dangerous, so it declares that it needs the `write` permission and that it `mutates_state`. Draco will never retry a mutating tool on its own.

## 2. Run tools from a workload

`ToolRuntime` is what actually executes tool calls. An agent calls `tools.call(ctx, name, arguments)`; the `ctx` carries the policy for this run. Here a plain function stands in for the model and runs whatever tool calls it is handed:

```python
from draco.tools import ToolRuntime

tools = ToolRuntime(registry)

async def agent(ctx, calls):
    report = []
    for name, args in calls:
        result = await tools.call(ctx, name, args)
        report.append((name, result.status, result.data if result.ok else result.error.code.value))
    return report
```

Every call returns a `ToolResult`. Its `status` is `success`, `empty`, `pending`, `error` or `timeout`. A refused call is an `error` result with a code such as `denied`, **not** an exception, so your agent loop can show the refusal to the model and let it recover.

## 3. Allow tools explicitly

Tool access is **default-deny**. A run can use a tool only if its `Policy` lists the tool in `allowed_tools`. If you don't pass one, the run can use no tools at all. Add the policy when you start the run:

```python
from draco import Harness, Policy

h = Harness()
h.register("agent", agent)

result = await h.run("agent", [("to_fahrenheit", {"celsius": 100})],
                     policy=Policy(allowed_tools=frozenset({"to_fahrenheit"})))
```

## 4. Try to misbehave

Here is the whole program. It tries each way a call can go wrong and prints what Draco did.

<!-- doctest -->
```python
import asyncio
from pydantic import BaseModel, Field
from draco import BudgetLimits, Harness, Policy
from draco.tools import ToolRegistry, ToolRuntime, native_tool

registry = ToolRegistry()

class Convert(BaseModel):
    celsius: float = Field(ge=-273.15)

@native_tool(registry, name="to_fahrenheit", description="Convert Celsius to Fahrenheit", input_model=Convert)
async def to_fahrenheit(args, ctx):
    return {"fahrenheit": args.celsius * 9 / 5 + 32}

class Nothing(BaseModel):
    pass

@native_tool(registry, name="delete_records", description="Delete all records", input_model=Nothing,
             permissions={"write"}, mutates_state=True)
async def delete_records(args, ctx):
    return {"deleted": 3}

tools = ToolRuntime(registry)

async def agent(ctx, calls):
    """Stands in for a model: run each requested tool call and report what happened."""
    report = []
    for name, args in calls:
        result = await tools.call(ctx, name, args)
        report.append((name, result.status, result.data if result.ok else result.error.code.value))
    return report

async def main():
    h = Harness()
    h.register("agent", agent)
    both = frozenset({"to_fahrenheit", "delete_records"})

    async def attempt(label, calls, **options):
        r = await h.run("agent", calls, **options)
        print(f"{label}: {r.status} {r.output}")

    boil = [("to_fahrenheit", {"celsius": 100})]
    await attempt("allowed tool", boil, policy=Policy(allowed_tools=frozenset({"to_fahrenheit"})))
    await attempt("no allow-list", boil)
    await attempt("bad argument", [("to_fahrenheit", {"celsius": -500})], policy=Policy(allowed_tools=both))
    await attempt("unknown tool", [("format_disk", {})], policy=Policy(allowed_tools=both | {"format_disk"}))
    wipe = [("delete_records", {})]
    await attempt("missing permission", wipe, policy=Policy(allowed_tools=both))
    await attempt("permission granted", wipe, policy=Policy(allowed_tools=both, permissions=frozenset({"read", "write"})))

    five = [("to_fahrenheit", {"celsius": c}) for c in range(5)]
    r = await h.run("agent", five, policy=Policy(allowed_tools=both), limits=BudgetLimits(max_tool_calls=2))
    print("tool budget:", r.status, r.error.message, "| tool calls charged:", r.usage.tool_calls)

    r = await h.run("agent", boil + wipe, policy=Policy(allowed_tools=both))
    print("trace:", [(e["event"], e["data"].get("tool")) for e in h.tracer.trace(r.execution_id)
                      if e["event"].startswith("tool.")])

asyncio.run(main())
```

<!-- output -->
```text
allowed tool: completed [('to_fahrenheit', 'success', {'fahrenheit': 212.0})]
no allow-list: completed [('to_fahrenheit', 'error', 'denied')]
bad argument: completed [('to_fahrenheit', 'error', 'invalid_input')]
unknown tool: completed [('format_disk', 'error', 'not_found')]
missing permission: completed [('delete_records', 'error', 'denied')]
permission granted: completed [('delete_records', 'success', {'deleted': 3})]
tool budget: budget_exhausted tool_calls budget exhausted (limit 2) | tool calls charged: 2
trace: [('tool.before', 'to_fahrenheit'), ('tool.after', 'to_fahrenheit'), ('tool.denied', 'delete_records')]
```

## 5. What each line shows

| Output | What happened |
|---|---|
| `allowed tool` | The tool is in `allowed_tools` and the run holds the permission it needs (`read` is the default), so it ran. |
| `no allow-list` | The same call with no `allowed_tools` is **denied**. Forgetting to configure access fails safe. |
| `bad argument` | `-500` fails the `ge=-273.15` rule. The handler never ran. The code is `invalid_input`. |
| `unknown tool` | A tool that isn't registered is `not_found`, even though the policy allowed its name. |
| `missing permission` | `delete_records` is allowed, but it needs `write` and the run only holds `read`. Denied. |
| `permission granted` | Grant `write` for this run and the same call succeeds. |
| `tool budget` | `max_tool_calls=2` lets two calls through. The third is not refused one call at a time: it **ends the whole run** as `budget_exhausted`, so a looping agent cannot keep going. |
| `trace` | The mutating call shows up as `tool.denied`. Denials are recorded as events, so you can audit them. |

Notice that in the first six lines the *run* is `completed` even when a tool call was refused. A refusal is information for the agent, not a crash. Only the budget, which is a hard limit on the run itself, changes the run's status.

## 6. Where the checks happen

When `tools.call` runs, Draco works through these steps in order and stops at the first refusal:

1. The tool exists. An unknown name is `not_found`.
2. The `Policy` allows the tool, the run holds the permissions it needs, and approval is granted if the tool requires it.
3. The harness's authorizer allows `tool.call` for this identity (and records an audit event).
4. Any critical `tool.before` hook can veto the call.
5. The tool-call **budget is charged**.
6. The arguments are validated against the tool's input model.
7. The concurrency limit and the timeout apply, and your handler runs.

Your handler is at the end of the list. That is the point: by the time your code runs, the call has passed every check.

Two consequences follow from the order. A call that is **denied costs nothing**, because it is refused before the budget is charged. A call that is **allowed but malformed costs one tool call**, because the budget is charged before the arguments are validated, so an agent that keeps sending bad arguments still runs into `max_tool_calls`.

See [the lifecycle](../concepts/lifecycle.md) and the [security model](../concepts/security.md).

## Next

- A tool that takes an argument naming a resource, such as a tenant or a document, needs one more check inside the handler. That is the [next tutorial](multi-tenant.md).
- [Add a tool](../guides/tools.md) covers approval-required tools, untrusted output and MCP.
