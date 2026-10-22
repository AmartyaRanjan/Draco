# Draco — Production-Oriented Agentic RAG

Draco is an agentic retrieval-augmented generation system built around hierarchical orchestration, controlled capability execution, evidence-grounded responses, and explicit context-budget management. It has two layers:

- **`app/` — the RAG/agent service.** A FastAPI application over Redis, Qdrant, OpenSearch and a vLLM server: PDF ingestion, hybrid retrieval with reranking, grounded chat, a tool-using agent with 21 capabilities, platform hardening, and an evaluation suite.
- **`src/draco/` — the execution harness package.** A reusable layer around the service: a harness with execution identity, policy, budgets, cancellation, tracing and persistence; a plugin/service host; an agent router; a guarded tool runtime (native tools, optional MCP); and resumable steps.

The service remains the first consumer of the package: no existing public API in `app/` changed, and the original test suite still runs unchanged.

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
   │ Docling + OCR      │          │ → ColBERT rerank   │         │ streaming · memory │
   │ chunking · BGE-M3  │          │ → thresholded      │         │ citations          │
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
- **Evaluation, tracing, and failure handling** — an 84-question RAG gold set and a 45-case agent suite; a redacted tracer with JSONL and OpenTelemetry exporters.

## Features and Implementation Status

| Capability | Status | Verification |
|---|---|---|
| Hierarchical agent orchestration | Implemented (supervisor + 21 capabilities) | `app/agent/`, [tools.md](tools.md) |
| Hybrid retrieval and reranking | Implemented (hit@5 0.79–0.97, faithfulness 0.95) | [evals.md](evals.md), [RETRIEVAL_LOG.md](RETRIEVAL_LOG.md) |
| Tool permissions and execution limits | Implemented, enforced at runtime | `tests/test_registry.py`, `tests/draco/unit/test_tools.py` |
| Context compaction | Implemented (summarisation + output-cap shrink) | `app/agent/`, [production.md](production.md) §2.10 |
| Persistent checkpoint recovery | Partial — harness `step()`/`resume()` and ingestion-job recovery work; agent graph checkpoints stay in memory | `tests/draco/reliability/`, `tests/test_agent_store_recovery.py`, `tests/test_ingestion_dispatch.py` |
| Tenant isolation | Partial — document ownership enforced on all retrieval paths; single-tenant trust domain otherwise | `tests/test_document_ownership.py`, [docs/DOCUMENT_OWNERSHIP.md](docs/DOCUMENT_OWNERSHIP.md) |
| Live external integrations | Implemented with approval gates and SSRF/injection defences; not exercised against live services in CI | [evals.md](evals.md), [production.md](production.md) §6 |

## Quick Start

Prerequisites: Python ≥ 3.11, Redis, Qdrant, OpenSearch, and a vLLM server (see `deploy/vllm/`). Full runbook: [production.md](production.md).

```bash
# RAG service
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
rqworker ingestion

# Draco harness package
pip install -e ".[dev]"
PYTHONPATH=src:. python -m examples.a_simple        # examples a..f in examples/
python -m pytest tests/draco -q                      # harness tests
python -m unittest discover -s tests -t .            # original suite, unchanged
```

Environment: copy and edit `.env.platform.example`, `.env.inference.example`, `.env.agent.example`. Configuration is validated at boot; `APP_ENV=production` refuses unsafe settings. Gate traffic on `GET /health/ready`.

## Example Execution Trace

Every request gets an execution id that is also the retrieval trace's request id. A grounded question follows: routing decision → capability plan → hybrid retrieval (evidence ids `[S1…]`) → context-budget check → model call → citation validation against the evidence store → streamed answer with citations. The tracer records each step (`h.tracer.trace(id)`, `h.tracer.tree(root_id)`), with secrets redacted and user/tenant ids hashed.

## Evaluation

Dataset, metrics, baselines and reproducibility instructions: [evals.md](evals.md). Headline numbers on 7 indexed PDFs (280 chunks): hit@5 0.79 as users write questions and 0.97 without the document-descriptor prefix, judged faithfulness 0.95, 8 of 8 unanswerable questions correctly declined, agent tool selection 0.71. Evaluation found two real agent defects (both fixed). These are not leaderboard-comparable numbers; failures and limits are documented.

## Deployment and Security

- Runbook, configuration reference and alerts: [production.md](production.md)
- Threat-scope: hardened for a **single-tenant trust domain**; document ownership now scopes retrieval per authenticated principal ([docs/DOCUMENT_OWNERSHIP.md](docs/DOCUMENT_OWNERSHIP.md))
- Security posture: API-key auth with brute-force lockout, Redis-backed token-bucket rate limits, LLM admission gate, request-size caps, PDF upload validation, default-deny tool policy, untrusted-output flagging, prompt-injection quarantine (a pattern filter, not a guarantee)
- Backups: Qdrant/OpenSearch snapshots plus the three SQLite files in `app/storage/`

## Known Limitations

- Prompt injection is mitigated, not solved; paraphrases can get past the filter ([evals.md](evals.md)).
- Retrieval is sensitive to phrasing; contextual chunk headers and query rewriting need re-indexing.
- The 4B model skips tools for some arithmetic; agent graph state is in-memory (a restart loses an in-flight run).
- Single-host data stores, no replication or automated backup; Docker image and CI workflow have not been built/run end-to-end.
- Previously committed runtime files remain in published Git history; treat them as exposed and rotate credentials.

## Roadmap

Planned work, separated from what is implemented: [docs/architecture/roadmap.md](docs/architecture/roadmap.md). Priorities: running CI on GitHub, LangGraph durable checkpoints with real `interrupt()` approvals, real ACL enforcement via `ExecutionContext`, collapsing the two agent loops onto the harness, and live-provider evals.

## Contributing and License

Development setup: `pip install -e ".[dev]"`, then run both test suites (`pytest tests/draco` and `python -m unittest discover -s tests -t .`); ruff and mypy are configured in `pyproject.toml`. Project structure and design principles: [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md); implementation log and verification matrix: [docs/IMPLEMENTATION_LOG.md](docs/IMPLEMENTATION_LOG.md).
