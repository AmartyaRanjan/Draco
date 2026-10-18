# Implementation log

Branch `draco-framework` (uncommitted working tree; nothing pushed). Resume from here: read this file, run the commands under "Verification", continue with "Remaining work".

## Verification (commands run on 2026-10-18 and their actual results)
| Command | Result |
|---|---|
| `python -m pytest tests -q` (system pytest + `.venv` site-packages) | **409 passed, 17 skipped** (baseline before changes: 328 passed, 15 skipped) |
| `python -m unittest discover -s tests -t .` (the original CI command) | Ran 343 tests, OK (skipped=15): unchanged |
| `python -m tests.draco.eval.runner` | routing 10/10, tools 10/10, termination 9/9, rag 7/7 |
| Clean venv, `pip install -e ".[dev]"` (only pydantic + httpx resolved; no torch/langgraph) | install OK; `ruff check` clean; `mypy src/draco` no issues (41 files); `pytest tests/draco/unit tests/draco/reliability`: 56 passed |
| Isolated venv with `mcp` 2.3.0 + `opentelemetry-sdk` | real MCP round trip, OTel span tree, MCP fake-client tests: 4 passed (these 2 tests skip in the main `.venv`, which lacks both) |
| `pip-audit -r requirements.txt --no-deps --disable-pip` | No known vulnerabilities found. (`pip-audit` on the throwaway venv flagged `pip` itself; CI upgrades pip first.) |
| `python -m examples.a_simple` ... `f_plugin` | all six exit 0; also asserted by `tests/draco/integration/test_examples.py` |

Not run: the GitHub Actions workflow, the Docker image, anything needing live vLLM/Qdrant/OpenSearch/Redis/GPU. Skipped tests are the repo's existing live-service tests plus the two optional-extra tests above.

## Deterministic vs provider-dependent
Deterministic (no model, network or randomness; nothing to seed): all `tests/draco`, the eval suites, the examples. They measure control-plane behaviour, not answer quality. Provider-dependent and not measured here: answer quality, latency, token counts (use the existing opt-in `scripts/eval_*`).

## What was added
- `pyproject.toml`, `.gitignore` (egg-info), `.github/workflows/ci.yml` (two jobs), `README.md` (section appended).
- `src/draco/` (41 modules): `core`, `plugins`, `policies`, `harness`, `observability` (+ `otel`), `state`, `agents`, `tools` (+ `mcp`), `adapters`.
- `tests/draco/{unit,integration,reliability,eval}` (pytest), `examples/` (a..f + offline fakes), `docs/` (architecture x3, API guide, extension guide, this log).
- `app/` was **not modified**.

## Decisions and trade-offs
1. `src/draco` alongside `app/` (no rename), built on the existing seams (see `current-state.md`). Cost: `app` ships as a top-level package in the same distribution (name collision risk before publishing).
2. Reuse over rebuild: tool results/errors/executor are the existing `ToolResult`/`CapabilityRegistry`; retrieval and agent runs are wrapped, not reimplemented. This makes `draco.tools` depend on three light `app.agent` modules (hence `httpx` in the base dependencies); an import-boundary test pins this.
3. One event bus for hooks, plugins, audit and tracing; critical handlers fail closed, others are isolated.
4. Harness returns results instead of raising for workload failures; typed errors are available through `raise_for_status()`.
5. Budget/depth/deadline/cancellation are enforced by the runtime, never by the model.
6. Recovery is step-level and honest: completed steps replay, unfinished non-idempotent steps demand reconciliation; no exactly-once claim.
7. User and tenant ids reach enforcing hooks raw but are hashed in traces and persisted events.

## Defects found and fixed during the work
- Tool tests used bare `BaseModel` as an input model, which pydantic cannot validate: added a `NoArgs` model.
- `execution.started` lacked user/tenant/session, so a quota plugin could not work: added, with PII hashing.
- `draco.tools` pulled `httpx` via `app.agent.web`: declared the dependency; mypy no longer follows into `app` (silent override).
- Legacy `unittest discover` would have failed importing pytest-style tests: `tests/draco/__init__.py` skips itself under unittest.
- App store contract found: `save_turn` needs `response_json["metadata"]["request_id"]`; documented and tied to the execution id.

## Definition-of-done matrix
| Criterion | Status | Evidence / caveat |
|---|---|---|
| Existing pipelines remain functional | **Verified** | 328 original tests unchanged (343 under unittest); `app/` untouched |
| Reusable harness executes existing and new workloads | **Verified** (with fakes) | `RetrievalService`, `AgentRunner`, LangGraph graph, plain callables; live backends not exercised |
| Plugins/services lifecycle and dependency semantics | **Verified** | `test_plugins.py`, rollback order, protected services |
| Routing and handoffs use validated contracts | **Verified** | `test_router.py`, eval `routing` 10/10; LLM strategy only with fake proposers |
| Tool calls validated, authorized, bounded, observable | **Verified** | `test_tools.py`; MCP against real SDK 2.3.0 in isolated venv |
| Execution context propagates across nested work | **Verified** | parent/child ids, correlation, shared budget, downward cancel |
| State/history persisted with a documented backend | **Verified** | SQLite + in-memory; conversation store reused (`test_conversation_adapter.py`) |
| Failures, timeout, cancel, budget have defined outcomes | **Verified** | `Status` enum, eval `termination` 9/9 |
| Recovery/retries avoid blind duplication of side effects | **Partially verified** | step replay, idempotency keys, reconciliation verified; `AgentRunner` runs are not resumable; external exactly-once not claimed |
| Concurrent executions isolated | **Partially verified** | in-process concurrency tested (contexts, SQLite, root concurrency limit); multi-process not tested |
| Core tests and runnable examples pass | **Verified** | 409 passed; examples run as tests |
| Packaging, install, docs, CI | **Partially verified** | clean-venv install, lint, types, audit verified; **CI workflow never run on GitHub**; Docker untouched |
| Optional integrations not mandatory | **Verified** | boundary test + clean venv without langgraph/torch/mcp/otel |
| Decisions and limitations documented | **Verified** | this file, `roadmap.md`, `current-state.md` |

## Known limitations
- Cancellation is cooperative and asyncio-based: work inside `asyncio.to_thread` (e.g. the ColBERT reranker) is abandoned, not interrupted.
- `ExecutionResult.usage` for a child is a budget-wide delta over its time window; concurrent siblings can inflate it. Root executions are exact.
- A `running` record from a dead process blocks its idempotency key until `resume()` is called (deliberate).
- Persisted `input` is JSON via `str()` fallback (lossy for non-JSON objects); every redacted event is stored with no retention policy.
- `requires_approval` tools are denied with `approval_required`; no approval workflow in Draco.
- MCP: single-page `list_tools` only; text content only; verified on one SDK version.
- Only Python 3.14 was exercised although `requires-python` says >=3.11.
- The reference study is partly *recalled*, not source-verified (see its header).

## Remaining work
See `docs/architecture/roadmap.md` (priority order: run CI, durable LangGraph checkpoints, resolve `app` package name, enforce ACLs, retention/Postgres store, unify agent loops, live evals).
