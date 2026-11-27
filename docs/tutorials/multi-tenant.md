# Tutorial: isolate tenants

**You will build** an agent that serves several customers ("tenants") from one process, and show that a user from one tenant cannot read another tenant's data, even when the model asks for it.

**You need** `pip install draco-harness`. This tutorial builds on [Guard a tool-using agent](guarded-tools.md): it uses the same tool pattern.

## The problem

You run one agent for customers Acme and Globex. Alice works at Acme. She asks the agent for "the notes for Globex". The model, being helpful, calls `read_notes(tenant="globex")`. Nothing in the prompt can reliably stop that. The model chose the argument, and the argument is just text.

Draco's answer has two parts:

1. **A trusted identity.** The caller's identity comes from your application, which has verified it, not from the request or the model.
2. **Authorization in code, against that identity.** A tool whose argument names a resource checks the resource's tenant against the identity's tenant before it touches data.

## 1. Tell Draco who is acting

An `Identity` has a subject (the user), a **mandatory tenant**, and optional roles. You build it after verifying the caller, for example from a validated login token. Draco ships no authentication; it trusts what you hand it:

```python
from draco import Identity

alice = Identity("alice", "acme", frozenset({"analyst"}))
```

Pass it to `run`. From then on, `user_id` and `tenant_id` come from the identity, children of this run inherit it, and nothing in the input or the arguments can replace it:

```python
result = await h.run("agent", "acme", identity=alice)
```

## 2. Check the resource the model named

When a tool argument names something that belongs to a tenant, authorize it inside the handler. `Resource` says what is being accessed and which tenant owns it, and `authorize` checks it against the trusted identity on the context:

```python
from draco.security import Resource, authorize

@native_tool(registry, name="read_notes", description="Read a tenant's notes", input_model=ReadNotes)
async def read_notes(args, ctx):
    await authorize(ctx, "notes.read", Resource("notes", args.tenant, tenant_id=args.tenant))
    return {"notes": NOTES[args.tenant]}
```

A tenant-scoped resource is reachable **only from the same tenant**. If `args.tenant` is `"globex"` and Alice's tenant is `acme`, `authorize` raises, the tool returns a `denied` result, and the data is never read. Each check, allowed or denied, is also recorded as a `security.allowed` or `security.denied` event.

## 3. Turn on the strict mode

Draco's default authorizer, `RoleAuthorizer`, can do more than isolate tenants. Give it a table of which roles may do what, and switch on `require_identity` so that **no identity means no access**:

```python
from draco.security import RoleAuthorizer

h = Harness(authorizer=RoleAuthorizer(
    {"analyst": {"workload.*", "tool.*", "notes.*"}},   # role -> allowed actions ("*" at the end is a wildcard)
    require_identity=True,
))
```

Without `require_identity`, a run with no identity is allowed, which suits a single-tenant app where one process serves one customer. For anything multi-tenant, turn it on.

## 4. The whole program

<!-- doctest -->
```python
import asyncio
from pydantic import BaseModel
from draco import Harness, Identity, Policy
from draco.security import Resource, RoleAuthorizer, authorize
from draco.tools import ToolRegistry, ToolRuntime, native_tool

NOTES = {"acme": ["acme roadmap"], "globex": ["globex salaries"]}
registry = ToolRegistry()

class ReadNotes(BaseModel):
    tenant: str                       # chosen by the model, so untrusted

@native_tool(registry, name="read_notes", description="Read a tenant's notes", input_model=ReadNotes)
async def read_notes(args, ctx):
    # The argument names a resource. Check it against the *trusted* identity before touching data.
    await authorize(ctx, "notes.read", Resource("notes", args.tenant, tenant_id=args.tenant))
    return {"notes": NOTES[args.tenant]}

tools = ToolRuntime(registry)

async def agent(ctx, tenant):
    r = await tools.call(ctx, "read_notes", {"tenant": tenant})
    return r.data if r.ok else f"{r.error.code.value}: {r.error.message}"

async def main():
    h = Harness(authorizer=RoleAuthorizer({"analyst": {"workload.*", "tool.*", "notes.*"}}, require_identity=True))
    h.register("agent", agent)
    policy = Policy(allowed_tools=frozenset({"read_notes"}))
    alice = Identity("alice", "acme", frozenset({"analyst"}))

    r = await h.run("agent", "acme", identity=alice, policy=policy)
    print("own tenant:    ", r.output)
    r = await h.run("agent", "globex", identity=alice, policy=policy)
    print("other tenant:  ", r.output)
    r = await h.run("agent", "acme", policy=policy)
    print("no identity:   ", r.status, "|", r.error.message)
    r = await h.run("agent", "acme", identity=alice, tenant_id="globex", policy=policy)
    print("spoofed tenant:", r.status, "|", r.error.message)
    bob = Identity("bob", "acme", frozenset({"intern"}))
    r = await h.run("agent", "acme", identity=bob, policy=policy)
    print("no role grant: ", r.status, "|", r.error.message)

asyncio.run(main())
```

<!-- output -->
```text
own tenant:     {'notes': ['acme roadmap']}
other tenant:   denied: 'notes.read' on notes 'globex' denied: cross-tenant access denied
no identity:    failed | 'workload.run' on workload 'agent' denied: no identity established
spoofed tenant: failed | user_id/tenant_id contradict the authenticated identity
no role grant:  failed | 'workload.run' on workload 'agent' denied: no role grants 'workload.run'
```

## 5. Reading the results

| Case | What Draco did |
|---|---|
| **own tenant** | Alice's tenant matches the resource's tenant and her role grants the actions. The tool returned data. |
| **other tenant** | The model asked for Globex's notes. `authorize` refused inside the handler, so the tool returned a `denied` result and **the run still completed**. The agent can tell the model "no" and carry on. |
| **no identity** | `require_identity=True` refused to start the run at all. The run's status is `failed`, and no workload code executed. |
| **spoofed tenant** | The caller passed `tenant_id="globex"` alongside Alice's identity. A tenant that contradicts the identity fails the run: the identity is the only source of truth. |
| **no role grant** | Bob is in the right tenant but his role (`intern`) is granted nothing, so even starting the workload is denied. |

Two details deserve a second look:

- **Where the checks run.** The first denial happens in your handler, so the run continues. The others happen before your workload starts. In both cases the protected effect, reading the notes, never happened.
- **Foreign ids look like missing ids.** Resuming or cancelling another tenant's run behaves as if the run does not exist, so one tenant cannot probe for another's execution ids.

## What Draco does not do

Draco does **not** authenticate anyone. If your application hands it a wrong `Identity`, Draco will faithfully enforce the wrong rules. Verify credentials first, build the `Identity` from the result, and never from a request field. Without any identity, `user_id` and `tenant_id` are plain strings from a trusted in-process caller and prove nothing.

Draco also does not make a shared index safe by itself. A retrieval backend that mixes tenants can return documents the caller must not see, so the retrieval adapter authorizes each document **before and after** the search. See [wrap a retrieval backend](../guides/retrieval.md).

## Next

- [Security model](../concepts/security.md) lists everything Draco enforces and everything it leaves to you.
- [Recover from a crash](recovery.md) is the third tutorial: durable steps and resume.
