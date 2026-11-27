# Security model

Draco enforces access control in code, at the point where an action would happen, never in the prompt. This page says exactly what that covers, and just as clearly what it leaves to you.

## The three layers

| Layer | Question it answers | Set by |
|---|---|---|
| **Identity** | Who is acting, and for which tenant? | Your application, after it verifies the caller |
| **Policy** | What may *this run* do: which tools, permissions and agents? | You, per run |
| **Authorizer** | May this identity do this action on this resource? Includes tenant isolation and role grants. | You configure it; Draco calls it |

They stack. A tool call must pass the policy **and** the authorizer, and a tool whose argument names a resource should additionally authorize that resource inside its handler.

## What Draco enforces

**Identity is the only trusted caller record.** `Identity(subject, tenant_id, roles)` rejects empty or placeholder values such as `anonymous`, `guest` or `*`. Given an identity, `user_id` and `tenant_id` are derived from it, and a contradicting `user_id=` or `tenant_id=` fails the run. Children inherit it, cannot replace it, cannot change user or tenant, and cannot carry trusted field names in `metadata`.

**Tool access is default-deny.** With no `allowed_tools`, a run can use no tools and sees none. `denied_tools`, permissions and approval requirements are checked again at execution time, so reaching a handler any other way does not bypass them.

**Children can only narrow.** A child run's `Policy` must grant nothing its parent does not. The same applies to `resume`: it restores the policy the run started under and refuses a more permissive one.

**The authorizer runs before the effect.** Draco calls it before each protected operation:

| Action | When |
|---|---|
| `workload.run` | Every execution |
| `execution.resume` | Every resume |
| `tool.call` | Every tool call |
| `retrieval.query` | Before a retrieval search |
| `document.read` | Before **and after** retrieval, because shared indexes over-return |
| `document.ingest` | Before ingestion |

The default `RoleAuthorizer` applies tenant isolation first: a tenant-scoped `Resource` is reachable only from the same tenant, and no identity means denial. Then, if you gave it a grant table, it checks roles. Any exception inside an authorizer, or a malformed decision, counts as a **denial**.

**Persistence is tenant-aware.** Idempotency keys live in a per-tenant namespace. `resume()` and `cancel()` of another tenant's execution behave as if the execution does not exist, so ids cannot be probed.

**Every decision is audited.** Each check emits `security.allowed` or `security.denied` with ids and a reason. `subject`, `user_id`, `tenant_id` and `resource_tenant` are pseudonymised in traces, and secrets are redacted ([tracing and redaction](observability.md)).

**Secrets never print.** `Secret` hides its value from `repr`, `str`, logging and JSON encoding, and refuses to be pickled. Reading it takes an explicit `.reveal()`.

## What you must do

!!! warning "Draco does not authenticate anyone"
    It trusts the `Identity` you pass. Verify credentials yourself (a signed token, a session, mTLS) and build the `Identity` from the verified result. Never build it from a request field the user controls.

- **Authenticate.** Without an `identity=`, `user_id` and `tenant_id` are plain strings from a trusted in-process caller. They prove nothing.
- **Turn on `require_identity` for multi-tenant use.** The default `RoleAuthorizer()` with no grants allows runs that have no identity, which suits a single-tenant deployment only. Set `require_identity=True` (or `DRACO_REQUIRE_IDENTITY`), or configure grants, and a missing identity is denied.
- **Authorize resources named by tool arguments.** Arguments come from a model and are untrusted. Inside the handler, call `authorize(ctx, action, Resource(...))`. See the [multi-tenant tutorial](../tutorials/multi-tenant.md).
- **Give retrieval and ingestion a document resolver.** Pass `document_resource=lambda doc_id: Resource("document", doc_id, tenant_id=...)` to the adapters. With an identity and no resolver, document access is **denied**, not allowed.
- **Share the rate limiter** if more than one process serves a tenant. The built-in one is per process ([rate limits](../guides/rate-limits.md)).
- **Encrypt state at rest** if the store holds sensitive inputs and outputs ([persistence](../guides/persistence.md)). Execution inputs and outputs are stored as given, because resume needs them.

## Which attack does what stop

| Attempt | What stops it |
|---|---|
| The model asks a tool for another tenant's data | The handler's `authorize` call denies a cross-tenant resource |
| The model calls a tool nobody allowed | Default-deny: no `allowed_tools` entry, no execution |
| A tool that writes runs under a read-only run | The tool's `permissions` are checked against the policy |
| A caller sends `tenant_id="other"` next to their own identity | The run fails: the claim contradicts the identity |
| A child agent tries to grant itself more tools | `Policy.narrows` refuses it |
| Another tenant resumes or cancels your run | Looks like "unknown execution" |
| A runaway loop of model or tool calls | Budgets end the run ([budgets](budgets.md)) |
| A secret in an event or a trace | Redaction removes it before storage and export |
| A tool returns text that tries to instruct the model | Output of `untrusted_output` tools is flagged and scanned. This is a **heuristic**; see below. |

## Known limits

- **Prompt-injection detection is a heuristic.** `looks_injected` flags suspicious tool output with patterns, and `untrusted_output=True` labels output as untrusted. Paraphrased attacks get past patterns. The real defence is that an injected instruction still cannot do anything the policy, the permissions and the authorizer do not allow. Treat the flag as a signal, not a guarantee.
- **Rate limits and the in-memory trace are per process.**
- **No authentication provider ships with Draco.**
- **Draco is not a sandbox.** A tool handler is your Python code and runs with your process's privileges. Draco decides whether it runs, not what it does once it does.

Found a vulnerability? Please do not open a public issue; follow the [security policy](https://github.com/AmartyaRanjan/Draco/blob/main/SECURITY.md).
