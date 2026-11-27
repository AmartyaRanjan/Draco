# Draco framework changelog

Newest first. The RAG service's own changelog moved to [draco-rag](https://github.com/AmartyaRanjan/draco-rag) with the service.

## Unreleased: the RAG service moves to its own repository; the package is now `draco-harness`

### Changed (packaging)
- The distribution is named **`draco-harness`** (the name `draco` on PyPI belongs to an unrelated project). Install with `pip install draco-harness[extra]`; the import name stays `draco`. Error messages and docs that suggested `pip install draco[...]` now say `draco-harness[...]`. Nothing was ever published under the old name, so there is no upgrade path to handle.
- Releases are built and published from GitHub Actions with PyPI trusted publishing (`.github/workflows/release.yml`).


### Removed (breaking)
- The RAG service (`app/`, its tests, scripts, Dockerfiles, `deploy/`, `requirements.txt` and its docs) moved to [AmartyaRanjan/draco-rag](https://github.com/AmartyaRanjan/draco-rag). This repository is now only the framework.
- The application-specific adapters moved with it, as `app.draco_adapters`: `draco.adapters.app_agent` (`agent_runner_workload`, `agent_resume_workload`), `app_chat` (`chat_workload`), `app_retrieval` (`retrieval_workload`, `rag_workload` over the app's `RetrievalService`), `app_ingestion` (`ingestion_workload`), `app_capabilities` (`register_app_capabilities`), `app_trace` (`AppLogExporter`), `conversation` (`sqlite_conversation_store`) and `errors` (`map_app_error`). `from draco.adapters import <name>` no longer resolves these names. The generic adapters (`draco.adapters.retrieval`, `ingestion`, `model_provider`, `providers`, `langgraph`, `langgraph_checkpoint`) are unchanged.
- The `app` extra is removed.
- The `rag` suite of the deterministic eval runner moved with the adapter it exercised.

### Changed
- `test_import_boundaries` now requires that no Draco module imports `app` (previously `draco.adapters` was exempt).
- Lower bounds restored to the oldest supported versions: `pydantic>=2.6`, `langgraph>=1.0` (a Dependabot group update had raised them to the newest releases). Dependabot now uses `versioning-strategy: increase-if-necessary`, so it no longer raises library lower bounds.
- CI: the `test` job installs every optional extra instead of the app stack, and the `docker-images` job moved to draco-rag. The tracked-data check also rejects `documents/` and `.pdf`/`.docx`/`.pptx`/`.ipynb` files.
- `scripts/check_history_exposure.sh` and `docs/HISTORY_PURGE.md` now also cover the personal documents and coursework once committed at the repository root, which the previous check missed.

## 0.3.0 (unreleased): state, limits and the unified request path

### Added
- **Retention**: `RetentionStore.purge(older_than=, events_older_than=)` on the in-memory, SQLite and Postgres stores; `apply_retention`, `DracoSettings.execution_retention_days` / `event_retention_days` (`DRACO_EXECUTION_RETENTION_DAYS`, `DRACO_EVENT_RETENTION_DAYS`) and `scripts/draco_retention.py`. Executions in `waiting` or `running`, and their descendants, are never deleted. Purging a completed execution also forgets its idempotency key.
- **PostgreSQL store**: `draco.state.postgres.PostgresStore` (extra `postgres`, psycopg 3), same schema and in-place migration as SQLite, sharing one SQL implementation. `DRACO_STATE_URL` (a URL containing a password is rejected; use `PGPASSWORD` or `.pgpass`). `Harness` now builds its store from settings (`draco.harness.open_store`) when none is passed; previously `state_path` was ignored.
- **Time-window rate limiting**: `draco.policies.ratelimit` (`RateLimiter` protocol, in-process `InMemoryRateLimiter`), `tenant_rate_per_minute/_burst`, `user_rate_per_minute/_burst`, `Harness(rate_limiter=)`. Only root executions are limited and idempotent replays are free. An excess run ends `budget_exhausted` with resource `tenant_rate` or `user_rate`, `retryable`, with `retry_after_s`.
- **Encryption at rest**: `draco.security.crypto.FernetCipher` (extra `crypto`, MultiFernet) for execution input/output/error, event data and step data; the key comes from a `SecretProvider` through `DRACO_STATE_ENCRYPTION_KEY` (the secret's name; comma-separated keys, newest first); `store.rotate_encryption()`. Ids, status, times, tenant/user/roles, usage and policy stay in plain text because lookups and retention query them.
- `scripts/draco_live_eval.py`: five eval suites (qa, grounded, routing, tools, termination) through the harness against `OpenAICompatProvider` or `AnthropicProvider`; skips cleanly without `DRACO_EVAL_PROVIDER`.
- `draco.adapters.app_chat.chat_workload` and `agent_resume_workload`: the starter app's `/chat`, `/agent/chat` and resume run as Draco workloads with one execution id.

### Fixed
- `ExecutionResult.raise_for_status()` re-raised every failure as a bare `DracoError`, losing the error's class, `code`, `retryable` and `details`. A router propagating a child agent's failure therefore turned a policy denial into a generic error and a retryable provider outage into a non-retryable one. It now rebuilds the original typed error (`PolicyViolation`, `ContractError`, `ProviderError`, ...), keeping `code`, `retryable` and `details`; budget, timeout and cancellation keep their details too. Found by building `draco-helpdesk` on the library; tests in `tests/draco/unit/test_raise_for_status.py` fail without the change.

### Changed
- `draco.tools.safety.looks_injected` is now the only copy; the app imports it (the parity test checks that).
- The starter app's agent tool calls run under an explicit tool policy; under Draco's default deny-all policy they are refused.

### Verification
- `pytest tests/draco`: 172 passed, 14 skipped without Postgres; with `DRACO_TEST_POSTGRES_URL` set the store contract suite also runs against a real `postgres:16-alpine` container (182 passed, 3 skipped).
- Python 3.11 and 3.12: `tests/draco/unit` and `reliability` pass in clean venvs. `ruff` and `mypy src/draco` clean.
- Not verified: a GitHub Actions run of the new matrix, live calls through the HTTP providers or the live eval script, multi-process rate limiting.

### Still open
- The rate limiter is per process; several processes serving one tenant need a shared implementation of the protocol.
- `rotate_encryption` re-encrypts in one transaction (offline maintenance). Plaintext written before encryption was enabled can linger in Postgres dead rows until `VACUUM FULL`, and in the SQLite WAL until `vacuum()`.
- The tracer keeps an in-memory trace per execution with no retention option; the app discards it after each top-level run.
- A built-in authentication provider.

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
- **LangGraph checkpoint adapter** (`draco.adapters.langgraph_checkpoint`): stable execution/thread configuration, optional SQLite saver factory, validated approval resume commands and interrupt helpers. LangGraph and `langgraph-checkpoint-sqlite` remain behind the `agents` extra.
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
- Time-window rate limiting, encryption at rest and the duplicated `looks_injected` were open in 0.2.0 and are resolved in 0.3.0; a built-in authentication provider is still open.
