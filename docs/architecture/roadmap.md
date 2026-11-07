# Roadmap

Status key: **done** = implemented and covered by tests that ran; **partial** = works with stated limits; **planned** = not started.

## Done
- Packaging (`pyproject.toml`, src layout, extras), ruff + mypy config, CI on GitHub (runs on every push; the Docker-image job and the Python 3.11/3.12 matrix were added later and have **not yet run on GitHub**).
- Core contracts: `ExecutionContext`, `ExecutionResult`/`Status`, typed `DracoError` tree, `DracoSettings`, `BudgetTracker`, `Policy`, `EventBus`.
- Plugin/service host with dependency ordering, rollback and protected services.
- Harness runner: policy, guardrails, hooks, deadlines, cancellation, budgets, idempotency keys, persistence, resume.
- Agent registry/router (explicit, rule, capability, optional LLM strategy), handoffs, delegation.
- Tool runtime over the existing `CapabilityRegistry`, native tools, optional MCP adapter (verified against `mcp` 2.3.0 in an isolated venv).
- Stores: in-memory, SQLite and **PostgreSQL** (`draco.state.postgres`, verified against a real `postgres:16` container) behind the same protocols, with **retention** (`purge`, `DRACO_*_RETENTION_DAYS`, `scripts/draco_retention.py`) that never deletes `waiting`/`running` executions.
- **Time-window rate limiting** per tenant and per user (`draco.policies.ratelimit`), in process; a `RateLimiter` protocol for a shared implementation.
- **Encryption at rest** for persisted input/output/error/event/step data (`draco.security.crypto`, Fernet with key rotation).
- Tracer with redaction, JSONL and OpenTelemetry exporters.
- Adapters: `RetrievalService`, `AgentRunner`, `ChatOrchestrator`, ingestion worker, any compiled LangGraph graph; OpenAI-compatible and Anthropic model providers.
- Durable LangGraph checkpoints: SQLite saver, application execution inspection/resume APIs, and approval-bound resumption with owner checks.
- Both HTTP chat paths (`/chat`, `/agent/chat`, plus resume) run as Draco workloads through one harness, with a shared execution id, policy, budget, tracing and `Idempotency-Key`. `/agent/chat/stream` streams steps, tool events and tokens.
- Owner-scoped document ACLs in retrieval; scoped, expiring API keys with an audit log.
- Starter app: long-term memory (Qdrant + OpenSearch, RRF), rolling summaries, contextual chunk headers, query rewriting, `.md/.txt/.html/.docx` ingestion, optional LLM entailment judge.
- Docker API image and isolated worker image build, and one PDF was ingested end to end through the worker image into throwaway Qdrant and OpenSearch.
- Examples, deterministic eval suites, and `scripts/draco_live_eval.py` (live-provider evals through the harness; tested against mock transports only).
- Python 3.11 and 3.12 verified for the framework tests.

## Partial
- Two control flows remain: `ChatOrchestrator` (its own ReAct loop) and the LangGraph `AgentRunner` share model calls, retrieval, tool governance and citation handling, but not the loop itself. A full merge would change the `/chat` response contract (tool names, trace statuses, `S#` citations). `EvidenceRegistry` and `EvidenceStore` are still separate.
- `AgentRunner` resumes waiting approvals from SQLite for up to 30 days, reconciles lost waiting-status writes, and can reclaim a five-minute-old resume claim only when the checkpoint still contains the approval interrupt. The HTTP API's tenant scope maps to the authenticated API-key user; it is not a general tenant identity provider.
- Framework approval remains workload-defined; the starter app persists source decisions and runs ingestion only after approval.
- The rate limiter and the tracer's in-memory trace are per process; the tracer has no retention option.
- Retrieval quality changes (chunk headers, query rewriting) are implemented and unit-tested but **not re-measured**: that needs a re-index of a real corpus (`scripts/reindex.py`) and the eval scripts against the live stack.

## Planned, in priority order
1. Re-index a real corpus, re-run `scripts/eval_rag.py`, and recalibrate the 0.55 ColBERT threshold; grow the agent eval set from 9 to 50+ cases.
2. Run the GitHub workflow with the new Docker and Python-matrix jobs; build the images with default CUDA torch and test GPU embeddings.
3. Deprecate `/chat` mode=agent in favour of `/agent/chat`, then re-implement rag mode on the LangGraph runner with the `ChatResponse` format kept.
4. A shared (Redis) rate limiter and a tracer retention option.
5. A real authentication provider (the app still uses a static API-key list with expiry and scopes).
6. Ingest approved web pages as HTML instead of rendering them to PDF.
7. Move to Qwen3-8B-AWQ when GPU headroom allows.
