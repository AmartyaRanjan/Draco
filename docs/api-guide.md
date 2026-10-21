# Draco API guide

```
pip install -e .            # core: pydantic + httpx
pip install -e ".[dev]"     # + pytest, ruff, mypy, pip-audit
pip install -e ".[mcp]"     # optional MCP adapter     (.[otel] for OpenTelemetry, .[agents] for LangGraph)
```
The RAG engine in `app/` keeps its own dependencies in `requirements.txt`; only `draco.adapters` modules that wrap retrieval/ingestion/agents need them.

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
- **Policy**: `Policy(permissions, allowed_tools, denied_tools, allowed_agents, allow_approval_required)`. `allowed_tools=None` exposes no tools.

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
Built on the existing `CapabilityRegistry` (so the 21 existing capabilities work unchanged). `native_tool(registry, name=, description=, input_model=, permissions=, requires_approval=, mutates_state=, parallel_safe=, timeout_s=, untrusted_output=)` registers an async function. `ToolRuntime(registry, services=app_services).call(ctx, name, args)` returns a `ToolResult`: policy and permission check, `tool.before` veto, budget charge, concurrency limit, serialization when `parallel_safe=False`, retries for retryable errors on **non-mutating** tools only, untrusted-output flagging (`looks_injected`), `tool.after` audit event. `rt.specs(ctx)` lists only tools the policy allows. MCP: `await register_mcp_tools(registry, client, server_name=, allow={...})`; an explicit allow-list is mandatory, descriptions are sanitized, outputs flagged untrusted, tools treated as mutating and gated by the `mcp` permission.

## RAG and agents (`draco.adapters`)
`retrieval_workload(service)`, `rag_workload(service)` (retrieve, then the service's own generator as a budgeted model call; abstains without a model call when there is no evidence), `agent_runner_workload(runner)`, `ingestion_workload()`, `graph_workload(compiled_graph)`. `app` exceptions map to typed errors (`RetrievalError`/`InferenceError`: 422 `ContractError`, 504 `ExecutionTimeout`, 429/5xx retryable `ProviderError`). The retrieval trace's `request_id` equals the execution id.

Sessions: `draco.adapters.conversation.sqlite_conversation_store(path)` returns the app's `SqliteConversationStore` unchanged (ownership is enforced per user; `save_turn` requires `response_json['metadata']['request_id']`, so pass `ctx.execution_id` to tie each turn to its execution).

## Observability (`draco.observability`)
The `Tracer` records every bus event, redacted (`api_key`/`token`/`password`/`secret`/`authorization` keys, bearer/`sk-` strings, long payloads; `user_id`/`tenant_id` replaced by a stable hash). `h.tracer.trace(id)`, `h.tracer.tree(root_id)`. Exporters: `MemoryExporter`, `JsonlExporter(path)` (also `DRACO_TRACE_PATH`), `OtelExporter(tracer_provider)`. The store also persists redacted events per execution.

## Migration
No existing public API changed: `app/` is untouched, the old `unittest discover -s tests -t .` command still runs the original 343 tests (the new `tests/draco` package is pytest-based and skips itself under unittest). New: `pip install -e .`, run new tests with `pytest tests/draco`.
