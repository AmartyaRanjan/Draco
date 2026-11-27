# Errors and statuses

## Execution statuses

`ExecutionResult.status` says how a run ended.

| Status | Meaning | Resumable with `resume`? |
|---|---|---|
| `completed` | The workload returned normally. | No |
| `failed` | The workload raised, or a check refused the run. `error` says why. | Yes |
| `cancelled` | The caller or the parent cancelled it. | Yes |
| `timeout` | The deadline passed. | Yes |
| `budget_exhausted` | A [budget, rate limit or concurrency cap](../concepts/budgets.md) was hit. | No |
| `handoff` | The workload returned a `Handoff` for the router to follow. | No |
| `waiting` | The workload returned `Waiting`: paused until `resume`. | Yes |

A run left `running` by a process that died is also resumable.

## Errors

Every Draco error is a `DracoError` with a stable `code`, a `message`, a `retryable` flag and a `details` dict. In an `ExecutionResult` they appear as `ErrorInfo` with the same fields plus `type`, the class name.

| Error | `code` | Retryable by default | Raised when |
|---|---|---|---|
| `ConfigurationError` | `configuration` | no | A setting is missing or invalid. The message names it. |
| `ContractError` | `contract` | no | Input or output failed validation, a name is registered twice, or `resume` is called on a run that cannot be resumed. |
| `RoutingError` | `routing` | no | Unknown workload or agent, no route found, a handoff loop, or too many hops. |
| `ToolError` | `tool` | no | A category for your own code. Draco itself reports tool failures as `ToolResult`s, not as this exception. |
| `ProviderError` | `provider` | set per error | A model or backend call failed. For the shipped HTTP providers, statuses 408, 409, 425, 429, 500, 502, 503, 504 and 529, and timeouts, are retryable; other statuses are not. |
| `PolicyViolation` | `policy` | no | Denied by the policy, the authorizer, tenant isolation or a guardrail. |
| `ApprovalRequired` | `policy` | no | A subclass of `PolicyViolation`: allowed only with an approval the policy does not grant. |
| `ExecutionTimeout` | `timeout` | **yes** | The deadline passed. |
| `ExecutionCancelled` | `cancelled` | no | Cancelled. |
| `PersistenceError` | `persistence` | **yes** | The state store failed. |
| `BudgetExhausted` | `budget` | no (rate limits: yes) | A limit was reached. `details` has `resource` and `limit`. |
| `PluginError` | `plugin` | no | A plugin could not be registered, started or stopped. |
| `DuplicateRegistration`, `MissingDependency`, `DependencyCycle` | `duplicate_registration`, `missing_dependency`, `dependency_cycle` | no | Plugin registration problems, found before anything runs. |
| `ReconciliationRequired` | `reconciliation_required` | no | A non-idempotent step started but never finished. See [state](../concepts/state.md). |

Any exception that is **not** a `DracoError` becomes `unexpected error (TypeName)` with code `internal` and `retryable=False`. The text of the original exception is not exposed.

`BudgetExhausted.details["resource"]` is one of `model_calls`, `tool_calls`, `tokens`, `retries`, `depth`, `tenant_concurrency`, `tenant_rate`, `user_rate` or `graph_steps`. Running out of time ends a run as `timeout`, not as `budget_exhausted`.

### Re-raising a result

```python
result.raise_for_status()      # raises the typed error behind a failed result; returns None if it completed
```

The raised exception has the same `code`, `retryable` and `details` as the result's `error`.

## Tool results

`ToolRuntime.call` returns a `ToolResult` instead of raising. Its `status` is one of:

| Status | Meaning |
|---|---|
| `success` | The handler returned data. |
| `empty` | The handler ran and found nothing. |
| `pending` | The handler started work that finishes later. |
| `error` | The call was refused or failed. `error.code` says why. |
| `timeout` | The call exceeded its `timeout_s`. |

When `status` is `error`, `error.code` is an `ErrorCode`. The first group is produced by the framework; the second group is vocabulary for **your tool handlers** to raise with `ToolFailure(code, message, retryable=...)`.

**Produced by Draco**

| Code | Meaning |
|---|---|
| `invalid_input` | The arguments failed validation. |
| `not_found` | The tool is not registered. |
| `denied` | The policy or the authorizer refused it. |
| `approval_required` | The tool needs an approval the policy does not grant. |
| `timeout` | The tool exceeded its `timeout_s`. |
| `output_too_large` | The output exceeded `max_output_chars`. |
| `internal` | The handler raised something unexpected, or returned an invalid result. The message is generic. |
| `upstream_error` | Also produced for a failing MCP tool. |

**Available to your handlers**

| Code | Suggested use |
|---|---|
| `upstream_error` | An outside service failed. Usually `retryable=True`. |
| `not_configured` | A dependency the tool needs is not set up. |
| `unsafe_url` | A URL was rejected as unsafe. |
| `limit_exceeded` | A limit inside the tool was exceeded. |
| `conflict` | The action conflicts with the current state. |

A refused tool call is a **result**, not a failed run. Only budget exhaustion, cancellation and the deadline end the run itself.

## Mapping to HTTP

Draco does not serve HTTP, but a service has to turn outcomes into status codes. [draco-rag](https://github.com/AmartyaRanjan/draco-rag) uses this mapping, which is a sensible starting point:

| Outcome | HTTP status |
|---|---|
| `timeout` | 504 |
| `budget_exhausted` | 429 (use `retry_after_s` for `Retry-After` on rate limits) |
| `PolicyViolation` | 403 |
| `ContractError` | 422 |
| `cancelled` | 503 |
