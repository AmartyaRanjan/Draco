# Roadmap

Status key: **done** = implemented and covered by tests that ran; **partial** = works with stated limits; **planned** = not started.

## Done (branch `draco-framework`)
- Packaging (`pyproject.toml`, src layout, extras), ruff + mypy config, CI jobs (workflow written; **not yet run on GitHub**).
- Core contracts: `ExecutionContext`, `ExecutionResult`/`Status`, typed `DracoError` tree, `DracoSettings`, `BudgetTracker`, `Policy`, `EventBus`.
- Plugin/service host with dependency ordering, rollback and protected services.
- Harness runner: policy, guardrails, hooks, deadlines, cancellation, budgets, idempotency keys, persistence, resume.
- Agent registry/router (explicit, rule, capability, optional LLM strategy), handoffs, delegation.
- Tool runtime over the existing `CapabilityRegistry`, native tools, optional MCP adapter (verified against `mcp` 2.3.0 in an isolated venv).
- SQLite + in-memory stores for history, events and step checkpoints.
- Tracer with redaction, JSONL and OpenTelemetry exporters.
- Adapters: `RetrievalService`, `AgentRunner`, ingestion worker, any compiled LangGraph graph.
- Durable LangGraph checkpoints: SQLite saver, application execution inspection/resume APIs, and approval-bound resumption with owner checks.
- Six runnable examples, four deterministic eval suites.

## Partial
- Session persistence: the app's `SqliteConversationStore` is re-exposed through `draco.adapters.conversation`, but no Draco test exercises it.
- `AgentRunner` resumes waiting approvals from SQLite for up to 30 days, reconciles lost waiting-status writes, and can reclaim a five-minute-old resume claim only when the checkpoint still contains the approval interrupt. The HTTP API's tenant scope currently maps to the authenticated API-key user; it is not a general tenant identity provider.
- Framework approval remains workload-defined; the starter app persists source decisions and runs ingestion only after approval.

## Planned, in priority order
1. Run the CI workflow on GitHub; fix whatever it finds.
2. Resolve the `app` top-level package name before publishing (rename or namespace).
3. Real ACL enforcement in `resolve_filters` using `ExecutionContext.user_id/tenant_id`.
4. Retention policy for persisted events; a SQL/Postgres store behind the same protocols.
5. Collapse the duplicate agent loops (`ChatOrchestrator` vs `AgentRunner`) onto the harness.
6. Live-provider evals under the harness (existing `scripts/eval_*` need the stack).
