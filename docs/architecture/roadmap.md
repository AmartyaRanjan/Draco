# Roadmap

Status key: **done** = implemented and covered by tests that ran; **partial** = works with stated limits; **planned** = not started.

## Done
- Packaging (`pyproject.toml`, src layout, extras), ruff + mypy config, CI on GitHub on every push: the full suite with all extras, plus a lean pydantic-only job on Python 3.11, 3.12 and 3.14.
- Core contracts: `ExecutionContext`, `ExecutionResult`/`Status`, typed `DracoError` tree, `DracoSettings`, `BudgetTracker`, `Policy`, `EventBus`.
- Plugin/service host with dependency ordering, rollback and protected services.
- Harness runner: policy, guardrails, hooks, deadlines, cancellation, budgets, idempotency keys, persistence, resume.
- Agent registry/router (explicit, rule, capability, optional LLM strategy), handoffs, delegation.
- Tool runtime over the existing `CapabilityRegistry`, native tools, optional MCP adapter (verified against `mcp` 2.3.0 in an isolated venv).
- Stores: in-memory, SQLite and **PostgreSQL** (`draco.state.postgres`, verified against a real `postgres:16` container) behind the same protocols, with **retention** (`purge`, `DRACO_*_RETENTION_DAYS`, `scripts/draco_retention.py`) that never deletes `waiting`/`running` executions.
- **Time-window rate limiting** per tenant and per user (`draco.policies.ratelimit`), in process; a `RateLimiter` protocol for a shared implementation.
- **Encryption at rest** for persisted input/output/error/event/step data (`draco.security.crypto`, Fernet with key rotation).
- Tracer with redaction, JSONL and OpenTelemetry exporters.
- Adapters: generic retrieval and ingestion workloads, any compiled LangGraph graph; OpenAI-compatible and Anthropic model providers. (The RAG service's own adapters live in [draco-rag](https://github.com/AmartyaRanjan/draco-rag).)
- Durable LangGraph checkpoints: SQLite saver, checkpoint config carrying the execution id and identity, and validated approve/reject resume commands.
- Examples, deterministic eval suites, and `scripts/draco_live_eval.py` (live-provider evals through the harness; tested against mock transports only).
- Python 3.11, 3.12 and 3.14 verified for the framework tests in CI.

## Partial
- The rate limiter and the tracer's in-memory trace are per process; the tracer has no retention option.

## Planned, in priority order
1. A shared (Redis) rate limiter and a tracer retention option.

The RAG service's roadmap (evaluation, chat-flow merge, Docker/GPU, models) is in [draco-rag](https://github.com/AmartyaRanjan/draco-rag/blob/main/docs/roadmap.md).
