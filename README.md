# 🐉 Draco

> An execution harness for RAG and agent systems, with a production-oriented starter app.

[![CI](https://github.com/AmartyaRanjan/Draco/actions/workflows/ci.yml/badge.svg)](https://github.com/AmartyaRanjan/Draco/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![Status: alpha](https://img.shields.io/badge/status-alpha-orange)

[📖 Documentation](docs/README.md) · [🤝 Contributing](CONTRIBUTING.md) · [🛡️ Security](SECURITY.md) · [🗺️ Roadmap](docs/architecture/roadmap.md) · [📜 Changelog](docs/DRACO_CHANGELOG.md)

---

## 🏗️ Overview

Draco is built in two distinct layers:

1. **`src/draco/` (The Framework)**  
   A lightweight Python package (hard dependency: `pydantic`) that runs any async workload under a unified lifecycle. It provides trusted identity, role-based authorization, default-deny tool policies, budgets, rate limits, deadlines, idempotency, redacted tracing, and durable state (in-memory, SQLite, or PostgreSQL). It includes an agent registry/router, a guarded tool runtime (native tools + optional MCP), resumable steps, and adapters for LangGraph, OpenAI-compatible, and Anthropic providers.

2. **`app/` (The Starter Service)**  
   A production-ready FastAPI RAG and agent service built *on top of* the framework. It features Redis, Qdrant, OpenSearch, and a vLLM server. Capabilities include multi-format ingestion (PDF, Markdown, text, HTML, DOCX), hybrid retrieval with reranking, grounded chat, a 21-capability LangGraph agent with human-approved ingestion, long-term memory, scoped API keys, Docker images, and a comprehensive evaluation suite.

> 💡 **Unified Execution:** The service is the framework's first consumer. Both chat paths (`/chat`, `/agent/chat`) and approval resumes run as Draco workloads through a single harness, maintaining one execution ID from the HTTP request down to the checkpoint.

---

## 🏛️ Architecture

```text
                         ┌────────────────────────────────────────────┐
 HTTP clients ── TLS ──► │ FastAPI platform (app/api, app/core)       │
                         │ auth + lockout · rate limits · admission   │
                         │ request caps · health · metrics · logs     │
                         └───────┬──────────────┬─────────────┬───────┘
                                 │              │             │
              ┌──────────────────┘              │             └───────────────┐
              ▼                                 ▼                             ▼
   ┌────────────────────┐          ┌────────────────────┐         ┌────────────────────┐
   │ Ingestion          │          │ Retrieval          │         │ Inference          │
   │ (app/ingestion,    │          │ (app/retrieval)    │         │ (app/inference)    │
   │  app/db)           │          │ dense + BM25 → RRF │         │ vLLM · Qwen3-4B-FP8│
   │ Docling · 5 formats│          │ → ColBERT rerank   │         │ streaming · memory │
   │ chunk hdrs · BGE-M3│          │ → thresholded      │         │ citations          │
   │ RQ worker + outbox │          │   selection        │         │ tool-policy checks │
   └────────┬───────────┘          └─────────┬──────────┘         └─────────┬──────────┘
            │                                │                              │
            ▼                                └──────────────┬───────────────┘
   ┌────────────────────┐                                   ▼
   │ Stores             │                        ┌────────────────────┐
   │ Qdrant (dense)     │                        │ Agent              │
   │ OpenSearch (BM25)  │                        │ (app/agent)        │
   │ SQLite registry    │                        │ LangGraph supervisor│
   │ (owner-scoped)     │                        │ 21 capabilities    │
   │ Redis (RQ, limits) │                        │ evidence + budgets │
   └────────────────────┘                        └─────────┬──────────┘
                                                           │
                         ┌─────────────────────────────────┘
                         ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ draco harness (src/draco)                                   │
   │ core: context · results · typed errors · budgets · policy   │
   │ harness: runner · hooks · guardrails · idempotency · resume │
   │ state: SQLite/Postgres · retention · encryption · limits    │
   │ agents: registry · router (rules/LLM) · handoffs            │
   │ tools: ToolRuntime over CapabilityRegistry · MCP adapter    │
   │ adapters: retrieval/RAG/agent/graph workloads               │
   │ observability: redacted tracer · JSONL/OTel exporters       │
   └─────────────────────────────────────────────────────────────┘
```

### 🔑 Core Subsystems
- **Agent Orchestration:** LangGraph supervisor graph (classify → plan → execute → budget check → summarize → evaluate → generate → validate citations) with handoff-loop protection.
- **Typed Capability Registry:** 21 capabilities with strict permissions, validation, timeouts, and error mapping. `ToolRuntime` enforces default-deny policy at execution time.
- **Hybrid Retrieval:** Parallel Qdrant (BGE-M3) and OpenSearch (BM25) search, exact `chunk_id` deduplication, Reciprocal Rank Fusion (k=60), ColBERT reranking, and token-budgeted context building with `[S#]` evidence labels.
- **Document Ownership:** Every document carries an `owner_id`. Retrieval and agent paths strictly intersect requested IDs with the authenticated principal's registry rows.
- **Evidence Provenance:** Citations are built from a run-scoped `EvidenceStore`, never from model-written IDs. Ingestion approval is human-gated.
- **Durable Ingestion:** Uploads create an outbox row in the same SQLite transaction as the document. A leased dispatcher publishes deterministic RQ job IDs with capped backoff, recovering interrupted jobs.
- **Unified Request Path:** `/chat`, `/chat/stream`, `/agent/chat`, `/agent/chat/stream` (SSE), and approval resumes share identity, policy, budget, tracing, `Idempotency-Key` replay, and typed HTTP errors (403, 429, 504).

---

## 🛠️ Draco as a Framework (`src/draco`)

Draco's native core needs only `pydantic`. Retrieval, ingestion, LangGraph, MCP, and OpenTelemetry are optional adapters.

### The Contract
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

### Architecture Layers
| Package | Responsibility | Dependencies |
|---|---|---|
| `draco.core` | `ExecutionContext`, `Identity`, `Policy`, budgets, results, typed errors, events | `pydantic` |
| `draco.harness` | Lifecycle: identity, auth, guardrails, execution, deadlines, persistence, `resume()` | |
| `draco.agents` | `AgentRegistry`, `Router`, handoffs, delegation, loop/depth limits | |
| `draco.tools` | `Tool`, `ToolRegistry`, `ToolRuntime` (policy, retries, untrusted output), MCP tools | `draco[mcp]` |
| `draco.security` | `Authorizer`, audit events, `Secret`, `FernetCipher` (encryption at rest) | `draco[crypto]` |
| `draco.state` | Execution/event/checkpoint stores (`InMemory`, `Sqlite`, `Postgres`); retention | `draco[postgres]` |
| `draco.observability`| `Tracer`, redaction, JSONL and OpenTelemetry exporters | `draco[otel]` |
| `draco.plugins` | Service registry, plugin host, input/output guardrails, rate limiting | |
| `draco.adapters` | Translation to external components (`model_provider`, `retrieval`, `langgraph`, `app_*`) | see below |

*Dependency direction is strictly enforced: layers only import downward, and only `draco.adapters` may import `app`.*

### ⚙️ Installation
*(Note: `uv pip` is recommended, but standard `pip` works identically)*
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

### 🔒 Security Model
Authenticate in your application and hand Draco an `Identity(subject, tenant_id, roles)`. User and tenant are derived from it, children inherit it, and **nothing** in input, tool arguments, or metadata can override it. Protected operations go through an `Authorizer` before any side effect and are audited. Tenant isolation covers resources, idempotency keys, resume, and cancel.  
👉 *Full details: [docs/api-guide.md#security](docs/api-guide.md)*

### 🔭 Observability & Persistence
- **Observability:** Every lifecycle step emits an event. The `Tracer` redacts secrets and pseudonymises IDs before storing/exporting. Telemetry handler failures never break a run.
- **Persistence:** `Harness(store=SqliteStore(path))` persists executions, redacted events, and `step()` checkpoints. `resume(execution_id)` re-runs under the same ID.
- **Retention:** Purges old finished executions and never touches running/waiting ones. Encryption at rest (`FernetCipher`) covers input, output, error, event, and step data.

### 🧪 Examples
Run via `PYTHONPATH=src:. python -m examples.<name>`
| Example | Description |
|---|---|
| `a_simple` | Native workload, trace, persisted status |
| `b_rag` | Starter-app RAG through the adapter (needs app stack) |
| `c_multi_agent` | Agent registry, capability routing, handoffs |
| `d_tool` | Native tool, permissions, audit |
| `e_recovery` | Steps, failure, resume |
| `f_plugin` | Plugin with a service and hooks |
| `g_secure_tenants`| Identity, tenant isolation, role grants (no starter app) |
| `h_langgraph` | LangGraph graph through the adapter, interrupt and resume |

---

## 📊 Features & Implementation Status

| Capability | Status | Verification |
|---|---|---|
| **Hierarchical agent orchestration** | ✅ Implemented (supervisor + 21 capabilities) | `app/agent/`, [tools.md](tools.md) |
| **Hybrid retrieval & reranking** | ✅ Implemented (hit@5 0.79–0.97, faithfulness 0.95) | [evals.md](evals.md), [RETRIEVAL_LOG.md](RETRIEVAL_LOG.md) |
| **Tool permissions & limits** | ✅ Implemented, enforced at runtime | `tests/test_registry.py`, `tests/draco/unit/test_tools.py` |
| **Context compaction** | ✅ Implemented (summarisation + output-cap shrink) | [production.md](production.md) §2.10 |
| **Persistent checkpoint recovery** | ✅ Implemented (SQLite persistence, stable execution ID resume) | `tests/test_agent_resume.py`, `tests/draco/reliability/` |
| **Unified harness path** | ✅ Implemented (control flows not yet fully merged) | `tests/test_execution_path.py` |
| **State: Postgres, retention, encryption** | ✅ Implemented in framework; rate limiter is per-process | `tests/draco/unit/test_stores.py` |
| **Memory, chunk headers, query rewriting** | ⚠️ Implemented & unit-tested; **not re-measured** | `tests/test_long_term_memory.py`, etc. |
| **Tenant isolation** | ⚠️ Partial (document ownership enforced; single-tenant trust domain) | [docs/DOCUMENT_OWNERSHIP.md](docs/DOCUMENT_OWNERSHIP.md) |
| **Docker images** | ✅ API and worker build in CI; small PDF ingested end-to-end (CPU) | `Dockerfile`, CI `docker-images` job |

---

## 🚀 Quick Start

**Prerequisites:** Python ≥ 3.11, Redis, Qdrant, OpenSearch, and a vLLM server (see `deploy/vllm/`). Full runbook: [production.md](production.md).

```bash
# 1. RAG Service
uv pip install -r requirements.txt
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
uv run rqworker ingestion

# 2. Or as containers (CPU torch shown; omit build arg for CUDA)
docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu -t rag-api .
docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu -f deploy/worker/Dockerfile -t rag-worker .

# 3. Draco Framework Development
uv pip install -e ".[dev,agents]"
PYTHONPATH=src:. python -m examples.a_simple
python -m pytest tests/draco -q                      # Framework tests
python -m unittest discover -s tests -t .            # Service tests
```

### 🌐 HTTP API (Starter App)
| Route | Scope | Purpose |
|---|---|---|
| `POST /documents`, `GET /documents/{id}` | upload / read | Upload `.pdf .md .txt .html .docx`; status; page diagnostics |
| `POST /query` | read | Retrieval, optionally with a grounded answer |
| `POST /chat`, `POST /chat/stream` | chat | Grounded chat; SSE streaming |
| `POST /agent/chat`, `POST /agent/chat/stream` | agent | Tool-using agent; SSE streaming |
| `GET /agent/executions/{id}`, `POST /.../resume` | agent | Inspect a run; approve or reject a waiting source |
| `POST /agent/sources/{id}/approve` | agent | Human-gated web-source ingestion |
| `GET/POST /memory`, `DELETE /memory` | chat | Long-term memory (501 unless `MEMORY_BACKEND=hybrid`) |
| `GET /health`, `GET /health/ready`, `GET /metrics` | none | Liveness, readiness, Prometheus metrics |

> 💡 **API Keys:** `APP_API_KEYS` entries look like `key:user[:scopes][:expires=YYYY-MM-DD]`. A plain `key:user` may do everything except `admin`. `Idempotency-Key` is accepted on chat routes.

---

## 📈 Evaluation
Dataset, metrics, baselines, and reproducibility instructions: [evals.md](evals.md).  
**Headline numbers** (7 indexed PDFs, 280 chunks): hit@5 **0.79** (user questions) / **0.97** (without document-descriptor prefix), judged faithfulness **0.95**, 8/8 unanswerable questions correctly declined, agent tool selection **0.71**. Evaluation found and fixed two real agent defects.

---

## 🛡️ Deployment & Security
- **Runbook:** Configuration reference and alerts in [production.md](production.md).
- **Threat-scope:** Hardened for a **single-tenant trust domain**; document ownership scopes retrieval per authenticated principal.
- **Security Posture:** Scoped/expiring API keys, audit log (`rag.audit`), brute-force lockout, Redis-backed token-bucket rate limits, LLM admission gate, request-size caps, upload validation, default-deny tool policy, and prompt-injection quarantine (pattern filter).
- **Backups:** Qdrant/OpenSearch snapshots plus all configured SQLite state files in `app/storage/`.

---

## ⚠️ Known Limitations
- Prompt injection is mitigated, not solved; paraphrases can bypass the filter ([evals.md](evals.md)).
- Chunk headers and query rewriting are implemented but **not yet re-measured**; existing indexes need `scripts/reindex.py` for headers to apply.
- The 4B model skips tools for some arithmetic. Two chat control flows (`ChatOrchestrator`, `AgentRunner`) still exist, though both run through the Draco harness.
- Single-host data stores, no replication or automated backup. GPU builds, large documents, and new CI jobs are untested. The starter app's SQLite state is not encrypted.
- Auth is an API-key list with scopes/expiry, not a full login system. Rate limits in Draco are per-process.
- ⚠️ **Historical Exposure:** Previously committed runtime files remain in published Git history. Treat them as exposed and rotate credentials. Purge runbook: [docs/HISTORY_PURGE.md](docs/HISTORY_PURGE.md). Check exposure with `scripts/check_history_exposure.sh`.

---

## 🗺️ Roadmap
Planned work, separated from what is implemented: [docs/architecture/roadmap.md](docs/architecture/roadmap.md).  
**Priorities:** Re-index a real corpus and re-run evals, run new CI jobs, finish merging the two chat flows, implement a shared rate limiter, and integrate a real authentication provider.

---

## 🤝 Contributing, Security & License
Contributions are welcome! Start with [CONTRIBUTING.md](CONTRIBUTING.md) for setup, CI checks, code/test standards, and PR guidelines.  
Questions and ideas go to [Discussions](https://github.com/AmartyaRanjan/Draco/discussions); see [SUPPORT.md](SUPPORT.md). Please read the [Code of Conduct](CODE_OF_CONDUCT.md).

🚨 **Found a vulnerability?** Do not open a public issue. Follow [SECURITY.md](SECURITY.md).

**Quick Development Loop:**
```bash
uv pip install -e ".[dev,providers]"
pytest tests/draco
ruff check src tests/draco examples
mypy src/draco
# If you changed app/:
python -m unittest discover -s tests -t .
```

Draco is released under the [MIT License](LICENSE).  
Project structure and design principles: [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) · Implementation log: [docs/IMPLEMENTATION_LOG.md](docs/IMPLEMENTATION_LOG.md) · Document index: [docs/README.md](docs/README.md).