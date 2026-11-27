# Configuration

Every framework setting is a field of `DracoSettings`, which is immutable and validated. Build one in code, or read it from the environment:

```python
from draco import DracoSettings, Harness

settings = DracoSettings(default_timeout_s=30, max_tool_calls=20)      # in code
settings = DracoSettings.from_env()                                    # from DRACO_* variables
h = Harness(settings)
```

`from_env()` reads one variable per field, named `DRACO_` plus the upper-cased field name. An empty variable counts as unset. Unknown fields are rejected, and an invalid value raises `ConfigurationError` that **names the variable**, for example `DRACO_MAX_TOOL_CALLS: Input should be greater than 0`.

`Harness()` with no arguments uses `DracoSettings()` (the defaults below), **not** the environment. Call `from_env()` yourself if you want environment configuration.

Counts and durations must be positive. List-valued settings are comma-separated in the environment.

## Limits

| Field | Environment variable | Default | Meaning |
|---|---|---|---|
| `default_timeout_s` | `DRACO_DEFAULT_TIMEOUT_S` | `120.0` | Deadline for a run when `timeout_s` is not passed. Also the default `max_duration_s` budget. |
| `max_model_calls` | `DRACO_MAX_MODEL_CALLS` | none | Model calls per run |
| `max_tool_calls` | `DRACO_MAX_TOOL_CALLS` | none | Tool calls per run |
| `max_tokens` | `DRACO_MAX_TOKENS` | none | Tokens per run, as reported by providers |
| `max_retries` | `DRACO_MAX_RETRIES` | none | Retries per run |
| `max_depth` | `DRACO_MAX_DEPTH` | `8` | Nesting of child runs |
| `max_concurrency` | `DRACO_MAX_CONCURRENCY` | `8` | Top-level runs executing at once in this process |

These become the default `BudgetLimits` for every run. Pass `limits=` to `run` to override them for one run. See [budgets and limits](../concepts/budgets.md).

## Admission control

| Field | Environment variable | Default | Meaning |
|---|---|---|---|
| `max_concurrency_per_tenant` | `DRACO_MAX_CONCURRENCY_PER_TENANT` | none | Top-level runs in flight for one tenant. Excess runs are rejected. |
| `tenant_rate_per_minute` | `DRACO_TENANT_RATE_PER_MINUTE` | none | Sustained runs per minute per tenant |
| `tenant_rate_burst` | `DRACO_TENANT_RATE_BURST` | the rate (at least 1) | Bucket size for the tenant limit |
| `user_rate_per_minute` | `DRACO_USER_RATE_PER_MINUTE` | none | Sustained runs per minute per user within a tenant |
| `user_rate_burst` | `DRACO_USER_RATE_BURST` | the rate (at least 1) | Bucket size for the user limit |

See [rate-limit tenants and users](../guides/rate-limits.md).

## State

| Field | Environment variable | Default | Meaning |
|---|---|---|---|
| `state_path` | `DRACO_STATE_PATH` | none | SQLite file. With neither this nor `state_url`, state is in memory. |
| `state_url` | `DRACO_STATE_URL` | none | `postgresql://` URL. Wins over `state_path`. A URL containing a password is rejected: use `PGPASSWORD` or `~/.pgpass`. |
| `state_encryption_key` | `DRACO_STATE_ENCRYPTION_KEY` | none | The **name** of a secret holding comma-separated Fernet keys, newest first. Needs a persistent store. |
| `execution_retention_days` | `DRACO_EXECUTION_RETENTION_DAYS` | none | Purge finished executions older than this (never `waiting` or `running`) |
| `event_retention_days` | `DRACO_EVENT_RETENTION_DAYS` | none | Purge events older than this, keeping execution records |

See [persist, encrypt and purge state](../guides/persistence.md).

## Security

| Field | Environment variable | Default | Meaning |
|---|---|---|---|
| `require_identity` | `DRACO_REQUIRE_IDENTITY` | `false` | When true, every protected operation without a trusted `Identity` is denied. Turn this on for multi-tenant use. |

See the [security model](../concepts/security.md).

## Tracing and redaction

| Field | Environment variable | Default | Meaning |
|---|---|---|---|
| `trace_path` | `DRACO_TRACE_PATH` | none | Append a JSONL trace to this file |
| `redact_keys` | `DRACO_REDACT_KEYS` | `key, token, password, secret, authorization` | Event keys containing any of these words are replaced with `<redacted>` |
| `pii_keys` | `DRACO_PII_KEYS` | `user_id, tenant_id, subject, resource_tenant` | Event keys whose values are replaced by a stable hash |
| `max_payload_chars` | `DRACO_MAX_PAYLOAD_CHARS` | `2000` | Longer strings in events are truncated |

See [tracing and redaction](../concepts/observability.md).

## Things that are not in `DracoSettings`

- **Model servers, vector stores and other backends.** Their settings belong to your application. Draco only provides the contracts and adapters.
- **Secrets.** Pass them as `Secret` values or through a `SecretProvider` (the default reads environment variables), never as plain settings.
- **Anything per-run** such as the policy, an identity or a deadline for one call. Those are arguments to `run`.
