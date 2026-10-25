# Draco framework changelog

Newest first. This file tracks the `src/draco` framework; the root `CHANGELOG.md` covers the ingestion pipeline.

## 0.2.0 (unreleased): standalone framework

### Added
- **Identity and authorization** (`draco.core.identity`, `draco.security`): trusted `Identity`, `Authorizer`/`RoleAuthorizer`, `Resource`, audited `authorize()`, `Secret`. Enforced before `workload.run`, `execution.resume`, `tool.call`, `retrieval.query`, `document.read`, `document.ingest`.
- **Native tools** (`draco.tools`): `Tool`, `ToolRegistry`, `ToolRuntime`, `execute_tool`, `ToolFailure`, MCP tools, with no dependency on `app`.
- **Provider-neutral adapters**: `model_provider` (`ModelProvider`, `generate`), `retrieval` (`Retriever`, `Evidence`, authorization before and after retrieval), `ingestion` (`ingest_workload`).
- **Concrete model providers** (`draco.adapters.providers`, extra `providers`): `OpenAICompatProvider` (OpenAI, vLLM, llama.cpp) and `AnthropicProvider`, both over `httpx`, with typed retryable/non-retryable errors that never include response bodies.
- `adapters/app_capabilities.py`: the starter app's 21 capabilities as Draco tools.
- **Persisted policy**: `ExecutionRecord.policy`; `resume()` restores it and refuses a more permissive one.
- **Per-tenant concurrency cap**: `DracoSettings.max_concurrency_per_tenant` (`DRACO_MAX_CONCURRENCY_PER_TENANT`); excess root executions end `budget_exhausted` (`tenant_concurrency`) immediately instead of queuing.
- `DracoSettings.require_identity` (`DRACO_REQUIRE_IDENTITY`).
- A one-time warning when a `tenant_id` is supplied without an `Identity`.
- Examples `g_secure_tenants`, `h_langgraph`; docs: Security, Adapters and Migration sections in `docs/api-guide.md`.
- CI: the `draco-core` job installs `.[dev,providers]` and builds and imports a non-editable wheel without `app`.

### Changed
- **Packaging**: the wheel contains only `draco*` (`app` is deployed from source); the only hard dependency is `pydantic`. New extras `providers`, `app`.
- `Harness.run(identity=)`: user/tenant derive from the identity and cannot contradict it; children inherit identity, cannot change user/tenant or widen `Policy` (`Policy.narrows`); metadata cannot carry trusted field names.
- Idempotency keys are tenant-scoped (`scope_key`); `resume()` and `cancel()` of another tenant's execution behave as "not found".
- `graph_workload` (LangGraph): interrupt becomes `Waiting`, `GraphRecursionError` becomes `BudgetExhausted("graph_steps")`, run metadata carries tenant/session ids.
- `ToolRuntime.specs()` hides approval-required tools unless the policy allows approval; native handlers re-check policy (merged from PR #9).
- `subject` and `resource_tenant` are pseudonymised in traces and persisted events.
- The agent adapter no longer substitutes `"anonymous"`.
- SQLite `executions` gained `tenant_id`, `user_id`, `roles_json`, `policy_json` (migrated in place when the database is opened).

### Removed / breaking
- `draco.tools.guarded` (use `draco.tools.ToolRuntime`; no `services=`/`evidence_factory=`). Handlers receive `(args, ExecutionContext)`. `CapabilityRegistry`/`CapabilityError` remain as aliases of `ToolRegistry`/`ToolFailure`; app `Capability` objects must go through `register_app_capabilities`.
- `ToolResult.error` is `ToolErrorInfo`.
- `Harness.cancel()` on a tenant-scoped execution needs `identity=`.
- Idempotency records written before this version do not match new keys.
- `pip install draco` no longer installs the `app` package.

### Verification
- Clean venv (pydantic + httpx only), non-editable install: package imports without `app`, torch or langgraph; 96 framework tests pass, 1 skipped (real LangGraph).
- Full dev environment: `pytest tests` (framework and starter app), `ruff`, `mypy src/draco`; the final counts are in the commit message.
- Not verified: a GitHub Actions run, live vLLM/Qdrant/OpenSearch/Redis, live calls to the HTTP providers (tested against `httpx.MockTransport` only).

### Still open
- The starter app's own LangGraph `AgentRunner` keeps its state in memory. `graph_workload` accepts any LangGraph checkpointer for your own graphs, but no durable checkpointer is wired into the app.
- Time-window rate limiting (only a concurrency cap exists), encryption at rest, and a built-in authentication provider.
- `looks_injected` is duplicated in `app/agent/web.py` and `draco/tools/safety.py` because the app image ships without `draco`; `tests/draco/unit/test_injection_parity.py` fails if they diverge.
