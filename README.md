# 🐉 Draco

> An execution harness for RAG and agent systems.

[![CI](https://github.com/AmartyaRanjan/Draco/actions/workflows/ci.yml/badge.svg)](https://github.com/AmartyaRanjan/Draco/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![Status: alpha](https://img.shields.io/badge/status-alpha-orange)

[📖 Documentation](https://amartyaranjan.github.io/Draco/) · [🤝 Contributing](CONTRIBUTING.md) · [🛡️ Security](SECURITY.md) · [🗺️ Roadmap](docs/architecture/roadmap.md) · [📜 Changelog](docs/DRACO_CHANGELOG.md)

---

## 🏗️ Overview

Draco is a lightweight Python package that runs any async workload under one lifecycle. Its only hard dependency is `pydantic`. It provides:
- **Access control:** trusted identity, role-based authorization and default-deny tool policies.
- **Limits:** budgets, rate limits and deadlines.
- **Reliability:** idempotency and resumable steps.
- **Tracing:** redacted traces.
- **Durable state:** in-memory, SQLite or PostgreSQL.

It includes an agent registry and router, and a guarded tool runtime for native tools and, optionally, MCP tools. It also ships adapters for LangGraph, OpenAI-compatible providers and Anthropic.

> 🧩 **Built on Draco:** [draco-rag](https://github.com/AmartyaRanjan/draco-rag) is a FastAPI RAG and agent service that runs every chat, agent and resume request as a Draco workload. It started out in this repository as the starter app and now lives on its own.

---

## 🛠️ The Contract

```python
Workload = Callable[[ExecutionContext, Any], Awaitable[Any]]
```
Anything matching this shape can be registered with the `Harness`:

```python
import asyncio
from draco import Harness, Identity

async def greeter(ctx, name):
    return f"hello {name}"

async def main():
    h = Harness()
    h.register("greeter", greeter)
    r = await h.run("greeter", "world", identity=Identity("alice", "acme"))
    print(r.status, r.output, r.execution_id)   # Status.completed hello world <id>

asyncio.run(main())
```
`run()` always returns an `ExecutionResult` (`completed` | `failed` | `cancelled` | `timeout` | `budget_exhausted` | `handoff` | `waiting`). Failures are never reported as success; `result.raise_for_status()` re-raises them as typed `DracoError`.

## 🏛️ Architecture Layers

| Package | Responsibility | Dependencies |
|---|---|---|
| `draco.core` | `ExecutionContext`, `Identity`, `Policy`, budgets, results, typed errors, events | `pydantic` |
| `draco.harness` | Lifecycle: identity, auth, guardrails, execution, deadlines, persistence, `resume()` | |
| `draco.agents` | `AgentRegistry`, `Router`, handoffs, delegation, loop/depth limits | |
| `draco.tools` | `Tool`, `ToolRegistry`, `ToolRuntime` (policy, retries, untrusted output), MCP tools | `draco-harness[mcp]` |
| `draco.security` | `Authorizer`, audit events, `Secret`, `FernetCipher` (encryption at rest) | `draco-harness[crypto]` |
| `draco.state` | Execution/event/checkpoint stores (`InMemory`, `Sqlite`, `Postgres`); retention | `draco-harness[postgres]` |
| `draco.observability`| `Tracer`, redaction, JSONL and OpenTelemetry exporters | `draco-harness[otel]` |
| `draco.plugins` | Service registry, plugin host, input/output guardrails, rate limiting | |
| `draco.adapters` | Translation to external components (`model_provider`, `providers`, `retrieval`, `ingestion`, `langgraph`) | see below |

*Dependency direction is enforced by tests: layers only import downward, and no Draco module imports application code.*

## ⚙️ Installation
The package is named **`draco-harness`**; you still `import draco`. Once a release is on PyPI:
```bash
pip install draco-harness                  # native framework (pydantic only)
pip install "draco-harness[agents,providers]"
```
From a clone, for development *(`uv pip` is recommended, but standard `pip` works identically)*:
```bash
uv pip install -e .              # Native framework
uv pip install -e ".[agents]"    # LangGraph adapter
uv pip install -e ".[providers]" # OpenAI-compatible / Anthropic model providers
uv pip install -e ".[postgres]"  # PostgreSQL state store
uv pip install -e ".[crypto]"    # Encryption at rest
uv pip install -e ".[mcp]"       # MCP tools
uv pip install -e ".[otel]"      # OpenTelemetry exporter
uv pip install -e ".[dev]"       # pytest, ruff, mypy
```

## 🔒 Security Model
Your application authenticates the caller and hands Draco an `Identity(subject, tenant_id, roles)`.
- User and tenant come from that identity. Child executions inherit it.
- Nothing in input, tool arguments or metadata can override it.
- Protected operations pass through an `Authorizer` before any side effect, and every decision is audited.
- Tenant isolation covers resources, idempotency keys, resume and cancel.

👉 *Full details: [docs/api-guide.md#security](docs/api-guide.md)*

## 🔭 Observability & Persistence
- **Observability:** Every lifecycle step emits an event. The `Tracer` redacts secrets and pseudonymises IDs before storing or exporting. A failing telemetry handler never breaks a run.
- **Persistence:** `Harness(store=SqliteStore(path))` persists executions, redacted events and `step()` checkpoints. `resume(execution_id)` re-runs under the same ID.
- **Retention:** Purges old finished executions and never touches running or waiting ones. Encryption at rest (`FernetCipher`) covers input, output, error, event and step data.

## 🧪 Examples
Run via `PYTHONPATH=src:. python -m examples.<name>`. They also run as tests.

| Example | Description |
|---|---|
| `a_simple` | Native workload, trace, persisted status |
| `c_multi_agent` | Agent registry, capability routing, handoffs |
| `d_tool` | Native tool, permissions, audit |
| `e_recovery` | Steps, failure, resume |
| `f_plugin` | Plugin with a service and hooks |
| `g_secure_tenants`| Identity, tenant isolation, role grants |
| `h_langgraph` | LangGraph graph through the adapter, interrupt and resume |

A RAG example over a real retrieval service lives in draco-rag (`examples/b_rag.py`).

---

## ⚠️ Known Limitations
- No authentication provider ships with Draco; you supply the `Identity`.
- The rate limiter and the tracer's in-memory trace are per process.
- Prompt-injection detection (`draco.tools.safety`) is a heuristic.
- **History rewritten (2026-11-27):** runtime data and personal documents that were committed earlier were removed from every branch. Clones and forks made before that date still hold the old history, and GitHub may still serve old commits through closed pull requests. Details: [docs/HISTORY_PURGE.md](docs/HISTORY_PURGE.md). To check a clone, run `scripts/check_history_exposure.sh`.

## 🗺️ Roadmap
Planned work, kept separate from what is implemented: [docs/architecture/roadmap.md](docs/architecture/roadmap.md).

---

## 🤝 Contributing, Security & License
Contributions are welcome! Start with [CONTRIBUTING.md](CONTRIBUTING.md) for setup, CI checks, code and test standards, and pull request guidelines.
Questions and ideas go to [Discussions](https://github.com/AmartyaRanjan/Draco/discussions); see [SUPPORT.md](SUPPORT.md). Please read the [Code of Conduct](CODE_OF_CONDUCT.md).

🚨 **Found a vulnerability?** Do not open a public issue. Follow [SECURITY.md](SECURITY.md).

**Quick development loop:**
```bash
uv pip install -e ".[dev,providers]"
pytest tests/draco
ruff check src tests/draco examples
mypy src/draco
```

Licensed under the [MIT License](LICENSE).
