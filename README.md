# Draco: an execution harness for RAG and agent systems, with a production-oriented starter app

[![CI](https://github.com/AmartyaRanjan/Draco/actions/workflows/ci.yml/badge.svg)](https://github.com/AmartyaRanjan/Draco/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![Status: alpha](https://img.shields.io/badge/status-alpha-orange)

[Documentation](docs/README.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Roadmap](docs/architecture/roadmap.md) · [Changelog](docs/DRACO_CHANGELOG.md)

Draco has two layers:

- **`src/draco/`: the framework.** A small Python package (hard dependency: `pydantic`) that runs any async workload under one lifecycle: trusted identity and role-based authorization, default-deny tool policy, budgets and rate limits, deadlines and cancellation, idempotency, tracing with redaction, and durable state (in-memory, SQLite or PostgreSQL, with retention and optional encryption at rest). It adds an agent registry and router, a guarded tool runtime (native tools, optional MCP), resumable steps, and adapters for LangGraph and for OpenAI-compatible and Anthropic model providers.
- **`app/`: a starter RAG and agent service built on it.** A FastAPI application over Redis, Qdrant, OpenSearch and a vLLM server: ingestion of PDF, Markdown, text, HTML and DOCX, hybrid retrieval with reranking, grounded chat, a tool-using LangGraph agent with 21 capabilities and human-approved ingestion, long-term memory, scoped API keys, Docker images, and an evaluation suite.

The service is the framework's first consumer: both chat paths (`/chat`, `/agent/chat`) and approval resumes run as Draco workloads through one harness, with a single execution id from the HTTP request down to the checkpoint.

Several things are implemented but not measured or exercised live; they are listed under [Known Limitations](#known-limitations) and in [production.md](production.md) §6.

## Architecture

```
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

Core subsystems:

- **Agent orchestration and conditional routing** — a LangGraph supervisor graph (classify → plan → execute → budget check → summarize → evaluate → generate → validate citations), plus the `draco.agents` registry and router (explicit, rule-based, capability and optional LLM strategies) with handoff-loop protection.
- **Typed capability registry and execution policies** — 21 capabilities with permissions, validation, timeouts, size caps and error mapping; the `draco.tools` `ToolRuntime` re-checks policy and permissions at execution time (default-deny tool access), so bypassing model-visible specs does not bypass policy.
- **Hybrid dense and lexical retrieval with reranking** — parallel Qdrant (BGE-M3) and OpenSearch (BM25) search, exact `chunk_id` deduplication, Reciprocal Rank Fusion (k=60), ColBERT rerank sharing the embedding backbone, a relevance threshold with a minimum-evidence fallback, and token-budgeted context building with `[S#]` evidence labels.
- **Document ownership on retrieval** — every document carries an `owner_id`; retrieval, status and agent document-access paths always intersect requested IDs with the authenticated principal's registry rows. Legacy rows are never guessed; a verified single-tenant migration exists (`scripts/assign_legacy_document_owner.py`).
- **Evidence provenance and citation validation** — citations are built from a run-scoped `EvidenceStore`, never from model-written ids; ingestion approval is human-gated.
- **Context-budget management and evidence summarization** — hierarchical summarisation under a token budget, with geometric output-cap shrink on context-window overflow.
- **Durable ingestion with recovery** — uploads create an outbox row in the same SQLite transaction as the document; a leased dispatcher publishes deterministic RQ job IDs with capped backoff and reconciles pending or lost queue jobs every 30 s, so interrupted ingestion jobs are recovered instead of stalling.
- **Model serving and reasoning-mode adapters** — vLLM serving Qwen3-4B-FP8 with per-request thinking mode and hermes tool-call parsing; `draco.adapters` maps app exceptions to typed harness errors.
- **One request path** — `/chat`, `/chat/stream`, `/agent/chat`, `/agent/chat/stream` (SSE) and approval resumes run as Draco workloads: shared identity, policy, budget, tracing, `Idempotency-Key` replay, and typed errors mapped to HTTP (policy 403, budget 429, deadline 504). The two loops (`ChatOrchestrator`, `AgentRunner`) share model calls, retrieval, tool governance and citation handling, but not yet their control flow.
- **Memory and context** — optional per-user long-term memory (Qdrant dense + OpenSearch BM25 with RRF, strictly user-scoped, `/memory` routes), rolling conversation summaries, contextual chunk headers, and query rewriting for the plain chat path. All off or rule-based by default; their retrieval-quality effect has not been re-measured.
- **Multi-format ingestion** — PDF, Markdown, text, HTML and DOCX, each with size limits (DOCX has a zip-bomb guard); `scripts/reindex.py` re-ingests existing documents when chunk headers are turned on.
- **Auth and operations** — API keys with scopes (`read`, `chat`, `agent`, `upload`, `admin`) and optional expiry, an audit log, retention for conversations and agent state with systemd and compose scheduling, and buildable API and worker Docker images.
- **Evaluation, tracing, and failure handling** — an 84-question RAG gold set and a 45-case agent suite; a redacted tracer with JSONL and OpenTelemetry exporters.

## Draco as a framework (`src/draco`)

Draco's native core needs only `pydantic`; retrieval, ingestion, LangGraph, MCP and OpenTelemetry are optional adapters. `app/` is a starter application reached only through `draco.adapters.app_*`. What changed and what is still open: [docs/DRACO_CHANGELOG.md](docs/DRACO_CHANGELOG.md).

### The contract

```python
Workload = Callable[[ExecutionContext, Any], Awaitable[Any]]
```

Anything with that shape (a function, an agent, a RAG pipeline, a compiled LangGraph graph through an adapter) can be registered with `Harness` and run:

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

`run()` always returns an `ExecutionResult` with one of `completed | failed | cancelled | timeout | budget_exhausted | handoff | waiting`. A failure is never reported as success; `result.raise_for_status()` re-raises it as a typed `DracoError` (with `retryable`).

### Architecture

| Package | Responsibility | Needs |
|---|---|---|
| `draco.core` | `ExecutionContext`, `Identity`, `Policy`, budgets, results, typed errors, events, settings | pydantic |
| `draco.harness` | Lifecycle: identity, authorization, guardrails, execution, deadlines/cancellation, normalization, persistence, `resume()`, `call_model`, `call_with_retry`, `step` | |
| `draco.agents` | `AgentRegistry`, `Router` (explicit, rule, capability and optional LLM strategies), handoffs, delegation, loop and depth limits | |
| `draco.tools` | `Tool`, `ToolRegistry`, `ToolRuntime` (policy, authorization, budget, retries, untrusted output), native tools, MCP tools | MCP: `draco[mcp]` |
| `draco.security` | `Authorizer`/`RoleAuthorizer`, `Resource`, `authorize()`, audit events, `Secret`, `FernetCipher` (encryption at rest, key rotation) | crypto: `draco[crypto]` |
| `draco.state` | Execution, event and step-checkpoint stores: `InMemoryStore`, `SqliteStore`, `PostgresStore`; retention (`purge`, `apply_retention`) that never deletes `waiting`/`running` executions | Postgres: `draco[postgres]` |
| `draco.observability` | `Tracer`, redaction, JSONL and OpenTelemetry exporters | OTel: `draco[otel]` |
| `draco.plugins`, `draco.policies` | Service registry, plugin host with dependency ordering, input/output guardrails, per-tenant and per-user rate limiting (`RateLimiter` protocol, in-process default) | |
| `draco.adapters` | Translation to external components. Neutral: `model_provider`, `retrieval`, `ingestion`. Framework: `langgraph`. Starter app: `app_*`, `conversation`, `errors` | see below |

Dependency direction is enforced by `tests/draco/unit/test_import_boundaries.py`: layers only import downward, and only `draco.adapters` may import `app`.

### Install

```
pip install -e .              # native framework
pip install -e ".[agents]"    # LangGraph adapter
pip install -e ".[providers]" # OpenAI-compatible / Anthropic model providers (httpx)
pip install -e ".[postgres]"  # PostgreSQL state store
pip install -e ".[crypto]"    # encryption at rest
pip install -e ".[mcp]"       # MCP tools
pip install -e ".[otel]"      # OpenTelemetry exporter
pip install -e ".[dev]"       # pytest, ruff, mypy
```
The starter-app adapters additionally need the app's stack (`pip install -r requirements.txt`, plus `.[app]`).

### Security model (summary)

Authenticate in your application and hand Draco an `Identity(subject, tenant_id, roles)`; user and tenant are derived from it, children inherit it, and nothing in input, tool arguments or metadata can override it. Protected operations (`workload.run`, `tool.call`, `retrieval.query`, `document.read`, `document.ingest`, `execution.resume`) go through an `Authorizer` before any side effect and are audited; unresolvable or failing authorization denies. Tenant isolation covers resources, idempotency keys, resume and cancel. Full list of what is enforced, what you must configure, and known gaps: [docs/api-guide.md#security](docs/api-guide.md).

### Observability

Every lifecycle step emits an event (`execution.*`, `route.decided`, `handoff`, `model.*`, `tool.*`, `retrieval.after`, `retry`, `step.*`, `security.*`). The `Tracer` redacts secrets and pseudonymises user/tenant ids before storing or exporting. `Harness(exporters=[JsonlExporter(path)])` or `DRACO_TRACE_PATH`; telemetry handler failures never break a run.

### Persistence, retention and resumption

`Harness(store=SqliteStore(path))` (or `PostgresStore(dsn)`, or `DRACO_STATE_PATH` / `DRACO_STATE_URL`) persists executions, redacted events and `step()` checkpoints. `resume(execution_id)` re-runs under the same id; completed steps return their recorded results, and an unfinished non-idempotent step raises `ReconciliationRequired` instead of repeating a side effect. Draco does not promise exactly-once effects, and cancellation is cooperative (it takes effect at the workload's next `await`).

Retention (`DRACO_EXECUTION_RETENTION_DAYS`, `DRACO_EVENT_RETENTION_DAYS`, `scripts/draco_retention.py`) purges old finished executions and never touches running or waiting ones. Encryption at rest (`FernetCipher`, key from a `SecretProvider`) covers input, output, error, event and step data; ids, status, times, tenant and usage stay plain because lookups and retention query them. Rate limits (`DRACO_TENANT_RATE_PER_MINUTE`, `DRACO_USER_RATE_PER_MINUTE`) end excess requests as retryable `budget_exhausted`; the default limiter is per process.

### Examples

`PYTHONPATH=src:. python -m examples.<name>`

| | |
|---|---|
| `a_simple` | native workload, trace, persisted status |
| `c_multi_agent` | agent registry, capability routing, handoffs |
| `d_tool` | native tool, permissions, audit |
| `e_recovery` | steps, failure, resume |
| `f_plugin` | plugin with a service and hooks |
| `g_secure_tenants` | identity, tenant isolation, role grants (no starter app) |
| `h_langgraph` | LangGraph graph through the adapter, interrupt and resume |
| `b_rag` | starter-app RAG through the adapter (needs the app stack) |

## Features and Implementation Status

| Capability | Status | Verification |
|---|---|---|
| Hierarchical agent orchestration | Implemented (supervisor + 21 capabilities) | `app/agent/`, [tools.md](tools.md) |
| Hybrid retrieval and reranking | Implemented (hit@5 0.79–0.97, faithfulness 0.95) | [evals.md](evals.md), [RETRIEVAL_LOG.md](RETRIEVAL_LOG.md) |
| Tool permissions and execution limits | Implemented, enforced at runtime | `tests/test_registry.py`, `tests/draco/unit/test_tools.py` |
| Context compaction | Implemented (summarisation + output-cap shrink) | `app/agent/`, [production.md](production.md) §2.10 |
| Persistent checkpoint recovery | Implemented — app agent graph and executions persist in SQLite; waiting approvals resume by stable execution id | `tests/test_agent_checkpoint.py`, `tests/test_agent_resume.py`, `tests/test_agent_store_recovery.py`, `tests/draco/reliability/` |
| One harness path for both chat flows | Implemented; the two control flows are not yet merged | `tests/test_execution_path.py`, [docs/api-guide.md](docs/api-guide.md) |
| State: Postgres, retention, encryption, rate limits | Implemented in the framework; Postgres verified against a real container; rate limiter is per process; the starter app's own SQLite state is not encrypted | `tests/draco/unit/test_stores.py`, `tests/draco/unit/test_ratelimit.py` |
| Memory, summaries, chunk headers, query rewriting, multi-format ingestion | Implemented and unit-tested; **not re-measured** (no re-index or eval run); HTML and DOCX conversion not run through Docling | `tests/test_long_term_memory.py`, `tests/test_context_headers.py`, `tests/test_query_rewrite.py`, `tests/test_ingestion_formats.py` |
| Tenant isolation | Partial — document ownership enforced on all retrieval paths; single-tenant trust domain otherwise | `tests/test_document_ownership.py`, [docs/DOCUMENT_OWNERSHIP.md](docs/DOCUMENT_OWNERSHIP.md) |
| Docker images | API and worker build in CI; one small PDF was ingested end to end through the worker (CPU) | `Dockerfile`, `deploy/worker/Dockerfile`, CI `docker-images` job |
| Live external integrations | Implemented with approval gates and SSRF/injection defences; not exercised against live services in CI | [evals.md](evals.md), [production.md](production.md) §6 |

Dependency direction is enforced by `tests/draco/unit/test_import_boundaries.py`: layers only import downward, and only `draco.adapters` may import `app`.

Prerequisites: Python ≥ 3.11, Redis, Qdrant, OpenSearch, and a vLLM server (see `deploy/vllm/`). Full runbook: [production.md](production.md).

```bash
# RAG service
pip install -r requirements.txt          # installs as written (docling 2.142 with pillow 13)
uvicorn app.main:app --host 0.0.0.0 --port 8000
rqworker ingestion

# or as containers (CPU torch shown; omit the build arg for CUDA torch)
docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu -t rag-api .
docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu -f deploy/worker/Dockerfile -t rag-worker .

# Draco framework
pip install -e ".[dev,agents]"
PYTHONPATH=src:. python -m examples.a_simple
python -m pytest tests/draco -q                      # framework tests
python -m unittest discover -s tests -t .            # service tests
```

### HTTP API (starter app)

| Route | Scope | Purpose |
|---|---|---|
| `POST /documents`, `GET /documents/{id}`, `GET /documents/{id}/pages` | upload / read | Upload `.pdf .md .txt .html .docx`; status; page diagnostics |
| `POST /query` | read | Retrieval, optionally with a grounded answer |
| `POST /chat`, `POST /chat/stream` | chat | Grounded chat; SSE streaming |
| `POST /agent/chat`, `POST /agent/chat/stream` | agent | Tool-using agent; SSE streaming |
| `GET /agent/executions/{id}`, `POST /agent/executions/{id}/resume` | agent | Inspect a run; approve or reject a waiting source |
| `POST /agent/sources/{id}/approve`, `/reject`, `/ingest`; `POST /agent/feedback`; `GET /agent/capabilities` | agent | Human-gated web-source ingestion, feedback, capability status |
| `GET/POST /memory`, `DELETE /memory`, `DELETE /memory/{id}` | chat | Your own long-term memory (501 unless `MEMORY_BACKEND=hybrid`) |
| `GET /health`, `GET /health/ready`, `GET /metrics` | none | Liveness, readiness, Prometheus metrics |

`APP_API_KEYS` entries look like `key:user[:scopes][:expires=YYYY-MM-DD]`; a plain `key:user` may do everything except `admin`. `Idempotency-Key` is accepted on the chat routes.

Environment: copy and edit `.env.platform.example`, `.env.inference.example`, `.env.agent.example`. Configuration is validated at boot; `APP_ENV=production` refuses unsafe settings. Gate traffic on `GET /health/ready`.

## Example Execution Trace

Every request gets an execution id that is also the retrieval trace's request id. A grounded question follows: routing decision → capability plan → hybrid retrieval (evidence ids `[S1…]`) → context-budget check → model call → citation validation against the evidence store → streamed answer with citations. The tracer records each step (`h.tracer.trace(id)`, `h.tracer.tree(root_id)`), with secrets redacted and user/tenant ids hashed.

## Evaluation

Dataset, metrics, baselines and reproducibility instructions: [evals.md](evals.md). Headline numbers on 7 indexed PDFs (280 chunks): hit@5 0.79 as users write questions and 0.97 without the document-descriptor prefix, judged faithfulness 0.95, 8 of 8 unanswerable questions correctly declined, agent tool selection 0.71. Evaluation found two real agent defects (both fixed). These are not leaderboard-comparable numbers; failures and limits are documented.

## Deployment and Security

- Runbook, configuration reference and alerts: [production.md](production.md)
- Threat-scope: hardened for a **single-tenant trust domain**; document ownership now scopes retrieval per authenticated principal ([docs/DOCUMENT_OWNERSHIP.md](docs/DOCUMENT_OWNERSHIP.md))
- Security posture: scoped, expiring API keys with an audit log (`rag.audit`) and brute-force lockout, Redis-backed token-bucket rate limits, LLM admission gate, request-size caps, upload validation per format, default-deny tool policy, untrusted-output flagging, prompt-injection quarantine (a pattern filter, not a guarantee)
- Backups: Qdrant/OpenSearch snapshots plus all configured SQLite state files in `app/storage/`, including the agent checkpoint file

## Known Limitations

- Prompt injection is mitigated, not solved; paraphrases can get past the filter ([evals.md](evals.md)).
- Chunk headers and query rewriting are implemented but not yet re-measured; existing indexes need `scripts/reindex.py` for the headers to apply.
- The 4B model skips tools for some arithmetic. Two chat control flows (`ChatOrchestrator`, `AgentRunner`) still exist, though both run through the Draco harness.
- Single-host data stores, no replication or automated backup. The Docker images build and ingest a small PDF end to end with CPU torch; GPU builds, large documents and the new CI jobs are untested. The starter app's SQLite state is not encrypted.
- Auth is an API-key list with scopes and expiry, not a login system. Rate limits in Draco are per process.
- Previously committed runtime files remain in published Git history; treat them as exposed and rotate credentials. Purge runbook: [docs/HISTORY_PURGE.md](docs/HISTORY_PURGE.md) — check exposure with `scripts/check_history_exposure.sh`.

## Roadmap

Planned work, separated from what is implemented: [docs/architecture/roadmap.md](docs/architecture/roadmap.md). Priorities: re-index a real corpus and re-run the evals, run the new CI jobs, finish merging the two chat flows, a shared rate limiter, and a real authentication provider.

## Contributing, security and license

Contributions are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md): setup, the checks CI runs, code and test standards, and how to open a pull request. Questions and ideas go to [Discussions](https://github.com/AmartyaRanjan/Draco/discussions); see [SUPPORT.md](SUPPORT.md). Please read the [Code of Conduct](CODE_OF_CONDUCT.md).

Found a vulnerability? Do not open a public issue. Follow [SECURITY.md](SECURITY.md).

Quick development loop: `pip install -e ".[dev,providers]"`, then `pytest tests/draco`, `ruff check src tests/draco examples` and `mypy src/draco`; add `python -m unittest discover -s tests -t .` if you changed `app/`. Project structure and design principles: [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md). Implementation log and verification matrix: [docs/IMPLEMENTATION_LOG.md](docs/IMPLEMENTATION_LOG.md). Every document is indexed in [docs/README.md](docs/README.md).

Draco is released under the [MIT License](LICENSE).
