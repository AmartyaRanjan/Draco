# Draco

**Draco is a small Python harness that runs agent and RAG code under rules you control.**

An agent is just an async function until it meets the real world: a user from one customer asks for another customer's data, a model loops and burns your token budget, a deploy kills a run halfway through a payment. Draco wraps the function in one lifecycle that handles those cases in code, not in the prompt:

- **Who is acting.** A trusted identity with a tenant, checked before anything runs.
- **What it may do.** Tools are default-deny. A run only gets the tools and permissions you grant.
- **How much it may use.** Budgets for model calls, tool calls, tokens, depth and time. Deadlines and cancellation.
- **What happened.** A redacted trace of every step, with secrets removed and user ids hashed.
- **Where it left off.** Durable state, so a crashed or paused run resumes under the same id without repeating finished work.

Draco is not a model SDK, a prompt library or a graph builder. It works with whatever you already use (plain functions, LangGraph graphs, any model provider) and adds the layer around them. The only hard dependency is `pydantic`.

<!-- doctest -->
```python
import asyncio
from draco import Harness, Identity

async def triage(ctx, ticket):
    # Any async function of (ctx, input) is a workload.
    return {"queue": "billing" if "invoice" in ticket else "general"}

async def main():
    h = Harness()
    h.register("triage", triage)
    result = await h.run("triage", "invoice 42 is wrong",
                         identity=Identity("alice", "acme"), timeout_s=5)
    print(result.status, result.output)

asyncio.run(main())
```

<!-- output -->
```text
completed {'queue': 'billing'}
```

`run` never raises for a workload failure. It returns an `ExecutionResult` whose `status` says how the run ended and whose `error` says why, so a failure cannot be mistaken for success.

## Where to go next

| If you want to... | Read |
|---|---|
| Install it and run something in five minutes | [Installation](getting-started/installation.md), [Quickstart](getting-started/quickstart.md) |
| Learn by building | [Guard a tool-using agent](tutorials/guarded-tools.md), [Isolate tenants](tutorials/multi-tenant.md), [Recover from a crash](tutorials/recovery.md) |
| Understand how it works and why | [How a run works](concepts/lifecycle.md), [Security model](concepts/security.md) |
| Do a specific job | [How-to guides](guides/tools.md) |
| Look something up | [Configuration](reference/configuration.md), [Errors and statuses](reference/errors.md), [Python API](reference/core.md) |

## Status

Draco is **alpha (0.1)**. The API can still change between minor versions, and the changelog says when it does. The [roadmap](architecture/roadmap.md) separates what is done, what works with stated limits, and what is only planned. Things worth knowing before you depend on it:

- It ships **no authentication**. You verify the caller and hand Draco an `Identity`. See the [security model](concepts/security.md).
- Rate limits and in-memory traces are **per process**. Several processes need a shared limiter ([rate limits](guides/rate-limits.md)).
- Prompt-injection detection is a **heuristic** that flags suspicious tool output. It is not a guarantee.
- Draco avoids re-running steps it knows finished. It does **not** promise exactly-once effects ([state](concepts/state.md)).

## Built on Draco

[draco-rag](https://github.com/AmartyaRanjan/draco-rag) is a FastAPI RAG and agent service that runs every chat, agent and resume request as a Draco workload. It started in this repository as the starter app and now lives on its own, as an example of an application built on the framework.

## Source and license

[github.com/AmartyaRanjan/Draco](https://github.com/AmartyaRanjan/Draco) · MIT License · `pip install draco-harness`
