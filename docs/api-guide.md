# Draco API guide

```
pip install -e .            # native framework: pydantic only
pip install -e ".[dev]"     # + pytest, ruff, mypy, pip-audit
pip install -e ".[mcp]"     # optional MCP tools      (.[otel] OpenTelemetry, .[agents] LangGraph, .[app] httpx for the starter app)
```
The starter RAG engine in `app/` keeps its own dependencies in `requirements.txt`; only the `draco.adapters.app_*` modules need them. Nothing outside `draco.adapters` imports `app` (enforced by `tests/draco/unit/test_import_boundaries.py`).

## 60 seconds
```python
import asyncio
from draco import Harness

async def greeter(ctx, name):                 # any async (ctx, input) -> output
    return f"hello {name}"

async def main():
    h = Harness()
    h.register("greeter", greeter)
    r = await h.run("greeter", "world", session_id="s1", user_id="alice")
    print(r.status, r.output, r.execution_id)
    print([e["event"] for e in h.tracer.trace(r.execution_id)])
asyncio.run(main())
```
Runnable versions of everything below are in `examples/` (`python -m examples.a_simple` ... `f_plugin`, with `PYTHONPATH=src:.`).

## Contracts (`draco.core`)
- **`ExecutionContext`**: `execution_id`, `parent_execution_id`, `session_id`, `agent_id`, `user_id`, `tenant_id`, `correlation_id`, `created_at`, `cancel` (token), `deadline`, `budget`, `policy`, `depth`, `services`. `ctx.child()` derives a nested context (same correlation/session/budget, new id, cancellation flows downward only). `ctx.check()` raises on cancel/deadline/duration. Never store request state in globals; use `ctx.metadata`.
- **`ExecutionResult`**: `status` is one of `completed | failed | cancelled | timeout | budget_exhausted | handoff | waiting`; plus `output`, `error` (`ErrorInfo`: type, code, client-safe message, `retryable`), `usage`, `handoff`, `children`, `replayed`. `raise_for_status()` re-raises as a typed error. Unknown exceptions are reduced to their type name.
- **Errors**: `DracoError` (`.retryable`, `.details`, `__cause__` preserved) with `ConfigurationError`, `ContractError`, `RoutingError`, `ToolError`, `PolicyViolation`, `ProviderError`, `ExecutionTimeout`, `ExecutionCancelled`, `PersistenceError`, `BudgetExhausted`, `PluginError` (+ `DuplicateRegistration`, `MissingDependency`, `DependencyCycle`).
- **Config**: `DracoSettings.from_env()` reads `DRACO_*` and raises `ConfigurationError` naming the variable. Provider settings stay in the existing `app` settings.
- **Budgets**: `BudgetLimits(max_model_calls, max_tool_calls, max_duration_s, max_depth, max_retries, max_tokens, max_concurrency)`; counted by `BudgetTracker`, shared by an execution and its children.
- **Policy**: `Policy(permissions, allowed_tools, denied_tools, allowed_agents, allow_approval_required)`. Tool access is default-deny: `allowed_tools=None` permits no tool executions and exposes no tools; an explicit allow-list is required. `denied_tools`, permissions, and approval requirements are checked again at execution time.

## Harness (`draco.harness`)
`Harness.run(target, input, *, parent, session_id, user_id, tenant_id, policy, limits, timeout_s, cancel, idempotency_key, correlation_id, metadata)` always returns an `ExecutionResult` (it does not raise for workload failures; it re-raises only if the *caller's* task is cancelled, after recording it).
- Hooks: `h.on("execution.started" | "execution.finished" | "execution.failed" | "execution.cancelled" | "execution.timeout" | "execution.budget_exhausted" | "model.before|after|failed" | "tool.before|after|denied" | "handoff" | "retry" | "step.*" | "route.decided", handler, critical=False)`. **Non-critical handlers cannot break a run** (errors recorded in `bus.handler_errors`). **Critical handlers run first and fail closed**; raise `PolicyViolation` to veto `execution.started` or `tool.before`.
- Guardrails: `add_input_guardrail`, `add_output_guardrail` (a crashing guardrail blocks).
- Model calls: `await call_model(ctx, fn, ..., provider=, model=, usage_of=)` charges the budget and emits events. `call_with_retry` retries only `retryable` errors and charges each retry.
- Cancellation: `h.cancel(execution_id)` or pass a `CancellationToken`; timeouts come from `timeout_s`/`DracoSettings.default_timeout_s`.
- Idempotency: a completed run with the same `idempotency_key` is returned (`replayed=True`), never re-executed. A key whose run is `running` (in flight or interrupted) is refused; use `resume()`.
- Recovery: `step(ctx, name, fn, idempotent=False)` records completion durably. `h.resume(execution_id)` re-runs under the same id; completed steps return their recorded value; a started-but-unfinished step runs again only if `idempotent=True`, otherwise `ReconciliationRequired`. Draco does **not** promise exactly-once effects; it avoids re-running steps it knows completed.
- Persistence: `InMemoryStore` (default) or `SqliteStore(path)`; failures are logged and the computed result is kept unless `Harness(strict_persistence=True)`.

## Agents (`draco.agents`)
`AgentRegistry(h).register(name, fn, capabilities={...})`; `Router(h, registry, strategies=[explicit(), rules(...), by_capability(), llm_proposal(fn)], default_agent=, max_handoffs=5)`. `router.run(RouteRequest(input, agent=, require=))` returns an `ExecutionResult` whose output has `output`, `path`, `decisions`. Every destination (including handoff targets and LLM proposals) is validated against the registry and policy before it runs; handoff loops and the hop limit raise `RoutingError`. `router.delegate(ctx, agent, input)` runs a child agent and re-raises its failure. LLM routing is optional and never required.

## Tools (`draco.tools`)
Native contracts (no `app` dependency): `Tool` (name, `input_model`, `handler(args, ctx)`, `permissions`, `mutates_state`, `requires_approval`, `parallel_safe`, `untrusted_output`, `timeout_s`, `max_output_chars`), `ToolRegistry`, `ToolResult` (`success | empty | pending | error | timeout`), `ErrorCode`, `ToolFailure`. `native_tool(registry, name=, description=, input_model=, ...)` registers an async function; raise `ToolFailure(ErrorCode.x, msg, retryable=...)` for expected failures. (`CapabilityRegistry`/`CapabilityError` remain as aliases.)

`ToolRuntime(registry).call(ctx, name, args)`: validates input, then enforces in this order: `Policy.authorize_tool` (allow/deny lists, permissions, approval), the harness **authorizer** (`tool.call`, audited), `tool.before` veto hooks, budget charge, concurrency limit, serialization when `parallel_safe=False`, timeout (also bounded by the execution deadline), retries for retryable errors on **non-mutating** tools only, untrusted-output flagging (`looks_injected`), and a `tool.after` event that records argument *names*, never values. A handler that raises `PolicyViolation` yields a `denied` result. `rt.specs(ctx)` lists only tools the policy allows (approval-required tools only when `allow_approval_required`). Native tool handlers re-check the policy, so reaching a handler without `ToolRuntime.call` does not bypass it. Tool arguments are untrusted: when an argument names a resource, authorize that resource inside the handler (`draco.security.authorize`), see `examples/g_secure_tenants.py`.

MCP: `await register_mcp_tools(registry, client, server_name=, allow={...})`; an explicit allow-list is mandatory, descriptions are sanitized, outputs flagged untrusted, tools treated as mutating and gated by the `mcp` permission.

Starter-app capabilities: `draco.adapters.app_capabilities.register_app_capabilities(tool_registry, app_registry, services=...)` exposes the 21 existing capabilities as Draco tools (the app executor still runs them; calls without `ctx.user_id` are refused).

## Security (`draco.security`, `draco.core.identity`)
**Enforced by Draco**
- `Identity(subject, tenant_id, roles)` is the only trusted caller record. Empty or placeholder values (`anonymous`, `guest`, `*` ...) are rejected. Pass it as `Harness.run(..., identity=)`; `user_id`/`tenant_id` are derived from it and a contradicting `user_id=`/`tenant_id=` fails the run. Children inherit it, cannot replace it, cannot change user/tenant, cannot widen the parent's `Policy` (`Policy.narrows`), and `metadata` cannot carry trusted field names.
- `Authorizer` (default `RoleAuthorizer`): tenant isolation first (a tenant-scoped `Resource` is only reachable from the same tenant; no identity means denial), then optional role grants (`{"analyst": {"tool.*"}}`). A grant table, `require_identity=True` / `DRACO_REQUIRE_IDENTITY`, an authorizer exception, or a malformed decision all deny. Checks run before the protected effect: `workload.run` (every execution), `execution.resume`, `tool.call`, `retrieval.query`, `document.read` (before *and* after retrieval, because shared indexes over-return), `document.ingest`.
- Persistence is tenant-aware: idempotency keys live in a per-tenant namespace; `resume()` and `cancel()` of another tenant's execution behave like "not found".
- Every decision emits `security.allowed` / `security.denied` on the event bus with ids and reason only. `subject`, `user_id`, `tenant_id` and `resource_tenant` are pseudonymised in traces and persisted events; keys named `*key/token/password/secret/authorization*` and bearer/`sk-` strings are redacted. `Secret` never renders or pickles its value.

**Requires application configuration**
- Authentication: build `Identity` from a verified credential (implement `Authenticator`); Draco ships none. Without `identity=`, `user_id`/`tenant_id` are plain strings from a trusted in-process caller, not proof of identity.
- Document ownership: pass `document_resource=lambda doc_id: Resource("document", doc_id, tenant_id=...)` to the retrieval/ingestion adapters. With an identity and no resolver, document access is denied, not allowed.
- Default `RoleAuthorizer()` without grants or `require_identity` allows identity-less, tenant-less operations (single-tenant use). Turn on `require_identity` for multi-tenant deployments.
- Known gaps: no per-tenant rate limits or quotas, no encrypted-at-rest state, `Policy` is not persisted (a resumed execution gets the policy passed to `resume()`), and prompt-injection detection is a heuristic.

## Adapters (`draco.adapters`)
Provider-neutral and dependency-free: `model_provider` (`ModelProvider` protocol, `ModelRequest/Response`, `generate(ctx, provider, req)` = budgeted, evented model call with token accounting), `retrieval` (`Retriever` protocol, `Evidence`, `retrieval_workload(retriever, document_resource=)`), `ingestion` (`ingest_workload(callable, resource_for=)`). Optional framework: `langgraph.graph_workload(graph, recursion_limit=)` (`pip install draco[agents]`): thread id = execution id, `__interrupt__` becomes `Waiting`, resume with `h.resume(id, input=Command(resume=...))`, `GraphRecursionError` becomes `BudgetExhausted("graph_steps")`; model calls inside a graph are budgeted only if nodes use `call_model`. Starter app: `app_retrieval` (`retrieval_workload`, `rag_workload`, both accept `document_resource`), `app_agent`, `app_ingestion`, `app_capabilities`, `app_trace`, `conversation`, `errors`. `app` exceptions map to typed errors (422 `ContractError`, 504 `ExecutionTimeout`, 429/5xx retryable `ProviderError`). The retrieval trace's `request_id` equals the execution id.

Sessions: `draco.adapters.conversation.sqlite_conversation_store(path)` returns the app's `SqliteConversationStore` (ownership enforced per user). Conversation memory is separate from execution history/checkpoints (`draco.state`).

## Observability (`draco.observability`)
The `Tracer` records every bus event, redacted (`api_key`/`token`/`password`/`secret`/`authorization` keys, bearer/`sk-` strings, long payloads; `user_id`/`tenant_id` replaced by a stable hash). `h.tracer.trace(id)`, `h.tracer.tree(root_id)`. Exporters: `MemoryExporter`, `JsonlExporter(path)` (also `DRACO_TRACE_PATH`), `OtelExporter(tracer_provider)`. The store also persists redacted events per execution.

## Migration
- `draco.tools.guarded` is gone: `ToolRuntime` moved to `draco.tools.execution` (re-exported from `draco.tools`) and no longer takes `services=`/`evidence_factory=`. Tool handlers now receive `(args, ExecutionContext)`; app capabilities go through `register_app_capabilities`. `CapabilityRegistry`/`CapabilityError` are aliases of `ToolRegistry`/`ToolFailure`, but app `Capability` objects are no longer accepted directly.
- `ToolResult.error` is `ToolErrorInfo` (was the app's `ToolError`; `draco.core.ToolError` is still the exception).
- `ExecutionRecord` gained `tenant_id/user_id/roles`; SQLite databases are migrated in place on open. Idempotency keys are now stored tenant-scoped (`scope_key`), so records written by older versions with a key will not match new runs.
- `Harness.cancel()` of a tenant-scoped execution needs `identity=`.
- `app_agent` no longer substitutes `"anonymous"` when `user_id` is missing; it fails with `PolicyViolation`.
- New audit events (`security.allowed`/`security.denied`) appear in traces.

`app/` is untouched. The old `unittest discover -s tests -t .` command still runs the original 343 tests (the new `tests/draco` package is pytest-based and skips itself under unittest). New: `pip install -e .`, run new tests with `pytest tests/draco`.
