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
- Six runnable examples, four deterministic eval suites.

## Partial
- Session persistence: the app's `SqliteConversationStore` is re-exposed through `draco.adapters.conversation`, but no Draco test exercises it.
- Resume works for `step()`-based workflows. `AgentRunner` runs are not resumable (its LangGraph checkpointer is still a per-request `InMemorySaver`).
- Approval: tools needing approval are denied with `approval_required`; Draco has no approval workflow of its own (the app's HTTP approval endpoints are unchanged).

## Planned, in priority order
1. Run the CI workflow on GitHub; fix whatever it finds.
2. LangGraph durable checkpointer + `interrupt` wired to `Waiting`/`resume`.
3. Resolve the `app` top-level package name before publishing (rename or namespace).
4. Real ACL enforcement in `resolve_filters` using `ExecutionContext.user_id/tenant_id`.
5. Retention policy for persisted events; a SQL/Postgres store behind the same protocols.
6. Collapse the duplicate agent loops (`ChatOrchestrator` vs `AgentRunner`) onto the harness.
7. Live-provider evals under the harness (existing `scripts/eval_*` need the stack).
