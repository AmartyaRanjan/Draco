# Contributing to Draco

Thank you for considering a contribution. Draco is a small execution framework (`src/draco`) and a starter RAG/agent service built on it (`app/`). This guide explains how the repository is organised, how to set it up, what a change has to pass before it is merged, and the conventions that keep the project honest and safe.

By taking part you agree to the [Code of Conduct](CODE_OF_CONDUCT.md). Security problems are **not** reported through issues; see [SECURITY.md](SECURITY.md).

## Contents

1. [Ways to contribute](#1-ways-to-contribute)
2. [Before you start](#2-before-you-start)
3. [How the repository is organised](#3-how-the-repository-is-organised)
4. [Development setup](#4-development-setup)
5. [Checks your change must pass](#5-checks-your-change-must-pass)
6. [Code standards](#6-code-standards)
7. [Tests](#7-tests)
8. [Security-sensitive changes](#8-security-sensitive-changes)
9. [Documentation and claims](#9-documentation-and-claims)
10. [Dependencies](#10-dependencies)
11. [Branches and commits](#11-branches-and-commits)
12. [Pull requests](#12-pull-requests)
13. [Review and merging](#13-review-and-merging)
14. [Working on specific areas](#14-working-on-specific-areas)
15. [Getting help](#15-getting-help)
16. [License](#16-license)

---

## 1. Ways to contribute

- **Report a bug** with a minimal reproduction, using the bug-report issue form.
- **Propose a feature** or an extension point, using the feature-request form.
- **Improve documentation**: wrong or unclear statements are bugs.
- **Add or sharpen tests**, especially regression tests for behaviour that is described but not covered.
- **Fix an issue**. Look for ones labelled `good first issue` or `help wanted`.
- **Review pull requests**. Careful review is as valuable as code.

Questions and design discussion belong in [GitHub Discussions](https://github.com/AmartyaRanjan/Draco/discussions), not issues.

## 2. Before you start

- **Open an issue first for anything non-trivial**: a new feature, a public API change, a change to security behaviour, or anything touching more than a few files. Agree on the approach before writing it; this avoids rework on both sides. Typos, small fixes and obvious bugs can go straight to a pull request.
- **Check what already exists.** [docs/architecture/roadmap.md](docs/architecture/roadmap.md) lists what is done, partial and planned. [production.md](production.md) section 6 lists known gaps. Your idea may already be tracked.
- **Keep changes focused.** One concern per pull request. Do not mix a refactor with a behaviour change.

## 3. How the repository is organised

```
src/draco/     the framework: harness, core contracts, agents/router, tools, security,
               state stores, observability, plugins, adapters
app/           the starter service: FastAPI, ingestion, retrieval, inference, agent
tests/draco/   framework tests (unit, integration, reliability, deterministic evals)
tests/         starter-app tests (original suite)
examples/      runnable examples; they also run as tests
scripts/       operational and evaluation scripts
deploy/        vLLM, worker and maintenance deployment files
docs/          API guide, extension guide, architecture, project overview
```

Rules that are enforced by tests, not just by convention:

- **Dependency direction.** Layers import downward only, and only `draco.adapters` may import `app`. See `tests/draco/unit/test_import_boundaries.py`.
- **The framework's only hard dependency is `pydantic`.** Everything else (`httpx`, LangGraph, MCP, OpenTelemetry, Postgres, cryptography) is an optional extra and must be imported lazily. Importing `draco` must not import `app`, `torch`, `langgraph`, `psycopg` or `cryptography`.
- **The wheel contains only `draco*`.** The `app` package is deployed from source.

If you are unsure where code belongs: reusable, application-independent behaviour goes in `src/draco`; anything specific to the bundled RAG service goes in `app/` and is reached through a `draco.adapters` module.

## 4. Development setup

You need **Python 3.11 or newer**. CI runs the framework on 3.11, 3.12 and 3.14 and the full service tests on 3.14.

```bash
git clone https://github.com/AmartyaRanjan/Draco.git
cd Draco
python -m venv .venv && source .venv/bin/activate
```

### Framework only (enough for most contributions)

```bash
pip install -e ".[dev,providers]"
```

This installs `pydantic` plus the dev tools (`pytest`, `ruff`, `mypy`, `pip-audit`) and `httpx` for the HTTP model providers. It does not need Redis, Qdrant, OpenSearch, a GPU or a model server.

### Optional extras

| Extra | Needed for |
|---|---|
| `agents` | the LangGraph adapter and durable checkpointer |
| `providers` | OpenAI-compatible and Anthropic providers (`httpx`) |
| `postgres` | `PostgresStore` (psycopg 3) |
| `crypto` | encryption at rest (`cryptography`) |
| `mcp` | MCP tools |
| `otel` | the OpenTelemetry exporter |

### Starter app (only if you change `app/`)

```bash
pip install -r requirements.txt     # large: torch, docling, transformers
pip install -e ".[dev,agents,providers]"
```

`requirements.txt` pins exact versions and installs as written. The app tests use fakes and do not need running services. Do not install CUDA wheels just to run tests; the CPU wheel index works:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
```

### Optional services for specific tests

- **PostgreSQL store.** The store contract suite runs against Postgres when `DRACO_TEST_POSTGRES_URL` is set, and is skipped otherwise. Use a throwaway container:

  ```bash
  docker run -d --name draco-test-pg -e POSTGRES_PASSWORD=test -p 127.0.0.1:55432:5432 postgres:16-alpine
  export DRACO_TEST_POSTGRES_URL=postgresql://postgres:test@127.0.0.1:55432/postgres
  python -m pytest tests/draco/unit/test_stores.py -q
  docker rm -f draco-test-pg
  ```
- **Docker images.** `docker build -t rag-api .` and `docker build -f deploy/worker/Dockerfile -t rag-worker .` (add `--build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu` for CPU-only).

> **Never point tests or experiments at shared or production services.** A `.env` file may contain URLs of real Qdrant, OpenSearch or Redis instances. Use throwaway containers on `127.0.0.1` for anything that writes.

## 5. Checks your change must pass

CI runs these on every pull request. Run them locally first.

```bash
# framework tests, lint and types
python -m pytest tests/draco -q
ruff check src tests/draco examples
mypy src/draco

# starter-app tests (only needed if app/ or its adapters changed)
python -m unittest discover -s tests -t .

# deterministic control-plane evals
python -m tests.draco.eval.runner
```

CI jobs and what each proves:

| Job | Proves |
|---|---|
| `test` | the service and framework suites pass with the app stack installed (Python 3.14, CPU torch) |
| `draco-core` (3.11, 3.12, 3.14) | the framework installs and works with **only** `pydantic` and `httpx`: lint, types, the lean test subset, a non-editable wheel that imports without `app`, and a dependency audit |
| `docker-images` | the API and worker images build and import |

**Run the lean subset before you push** if you touched framework code or its tests. A test that quietly needs an app-only dependency passes in a full environment and fails in `draco-core`:

```bash
python -m venv /tmp/lean && /tmp/lean/bin/pip install -e ".[dev,providers]"
/tmp/lean/bin/python -m pytest tests/draco/unit tests/draco/reliability -q
```

A test that needs an optional package must skip when it is absent (`pytest.importorskip("bs4")`), not fail.

**Dependency audit.** `draco-core` runs `pip-audit` over the installed environment and over `requirements.txt`. A new advisory against a pinned or bundled package fails the build. Fix it by upgrading, not by ignoring it. If you believe an advisory does not apply, say why in the pull request.

## 6. Code standards

- **Formatting and lint:** `ruff` is configured in `pyproject.toml` (line length 110, rules `E F I B UP`). CI lints `src`, `tests/draco` and `examples`. `app/` still has older findings; do not add new ones in the code you touch.
- **Types:** `mypy src/draco` must be clean. Public functions and classes in the framework are annotated.
- **Async:** workloads are `async def (ctx, input) -> output`. Do not block the event loop; run blocking I/O with `asyncio.to_thread`.
- **Errors:** raise the typed errors in `draco.core.errors` (`PolicyViolation`, `ContractError`, `ProviderError`, ...). Set `retryable` honestly: it drives retries and HTTP status mapping. Client-facing messages must not contain secrets, response bodies or internal paths.
- **Optional dependencies:** import them inside the function or module that needs them, and raise `ConfigurationError` with an actionable message when they are missing.
- **No global request state.** Per-request data lives on `ExecutionContext`. Module-level singletons make isolation and testing harder.
- **Comments explain why**, not what. Remove commented-out code.
- **Public API changes** need a docs update and a changelog entry (see section 9). Prefer additive changes; call out breaking ones loudly.

## 7. Tests

Every behaviour change comes with a test.

- **Framework tests** go in `tests/draco/` (`unit/`, `integration/`, `reliability/`, `eval/`). **App tests** go in `tests/`.
- **Bug fixes need a regression test that fails without the fix.** Prove it: revert the fix, watch the test fail, restore the fix. Say so in the pull request.
- **No live services in the default suite.** Use the in-repo fakes (`examples/_fakes.py`, `tests/agent_helpers.py`) and `httpx.MockTransport` for HTTP providers. Tests that require a real service must skip themselves with a clear reason.
- **Tests must be deterministic.** No sleeping on wall-clock time where a fake clock or an injected `now` works, no dependence on test order, no network.
- **Security behaviour is tested from the attacker's side**: a different tenant, a forged argument, a missing identity, an over-returning backend. See `tests/draco/unit/test_security.py` and `examples/g_secure_tenants.py` for the style.
- **Do not weaken or delete a test to make a change pass.** If a test is wrong, fix it and explain why.
- The examples in `examples/` run as tests; keep them working and runnable with `PYTHONPATH=src:. python -m examples.<name>`.

Live evaluation scripts (`scripts/eval_*.py`, `scripts/draco_live_eval.py`) need a running stack or a provider key and are opt-in. Report numbers only from runs you actually did, with the dataset and settings.

## 8. Security-sensitive changes

Draco enforces authorization, tenant isolation and tool policy in code, not in prompts. Changes in these areas get extra scrutiny:

- `draco.security`, `draco.core.identity`, `draco.core.policy`, `draco.tools`
- document ownership and retrieval filters
- authentication, API-key scopes and rate limiting in `app/core`
- anything that parses untrusted input: uploads, web pages, tool output

Principles to follow:

- **Fail closed.** No identity, no matching grant, an exception in an authorizer, or a malformed decision all deny.
- **Tool arguments and model output are untrusted.** If an argument names a resource, authorize that resource inside the handler against the trusted identity.
- **Default-deny tool access.** A tool runs only if the policy explicitly allows it.
- **Never log or return secrets**, bearer tokens, API keys or provider response bodies.
- **Do not widen permissions in a child execution.** Children may only narrow the parent's `Policy`.
- **Treat a foreign id like an unknown id**, so resource ids of other tenants cannot be probed.

If you find a vulnerability, **do not open a public issue or pull request.** Follow [SECURITY.md](SECURITY.md).

## 9. Documentation and claims

This project prefers precise statements over impressive ones. Documentation is part of the change.

- Update the relevant docs in the same pull request: [docs/api-guide.md](docs/api-guide.md) for framework behaviour, [docs/extension-guide.md](docs/extension-guide.md) for extension points, [production.md](production.md) for operations and known gaps, and the [roadmap](docs/architecture/roadmap.md) when you finish or add planned work.
- **Changelogs:** framework changes go in [docs/DRACO_CHANGELOG.md](docs/DRACO_CHANGELOG.md); starter-app and ingestion changes go in [CHANGELOG.md](CHANGELOG.md). Add an entry under the unreleased heading and mention behaviour changes explicitly.
- **Separate verified from unverified.** State what you ran and what you did not (for example "tested against `httpx.MockTransport`, not a live endpoint"). Do not claim a measured improvement you did not measure.
- **Update stale statements you find**, including in old docs. A wrong sentence is a bug.
- Link with relative paths. Keep code samples runnable.

## 10. Dependencies

- Prefer the standard library. A new runtime dependency needs a stated reason in the pull request.
- **Framework (`pyproject.toml`):** the only hard dependency is `pydantic`. New third-party libraries must be an optional extra.
- **App (`requirements.txt`):** exact pins. Change a pin only when you must (security fix, bug, needed feature), and check that the whole file still resolves:

  ```bash
  uv pip compile requirements.txt --python-version 3.14 -o /tmp/resolved.txt
  pip-audit -r requirements.txt --no-deps --disable-pip --desc
  ```
- Packages with native or heavy dependencies (`torch`, `docling`, `pillow`) interact. When a bump conflicts, find the smallest compatible upgrade and verify real behaviour (for example, convert a PDF in the worker image) rather than relying on the resolver alone.
- Dependabot opens update pull requests; they follow the same checks as any other change.

## 11. Branches and commits

- Branch from `main`. Use a descriptive prefix: `feat/`, `fix/`, `docs/`, `test/`, `ci/`, `chore/` (for example `fix/router-error-type`).
- **Commit messages** follow the style already in the history: a short imperative summary with a type prefix, then a body that explains **why**.

  ```
  fix(core): raise_for_status keeps the original error type

  A router re-raising a child agent's failure turned every error into a bare
  DracoError, so a policy denial surfaced as a generic error.
  ```

  Types in use: `feat`, `fix`, `docs`, `test`, `ci`, `chore`, `refactor`, with an optional scope such as `(core)`, `(agent)`, `(deps)`.
- Keep commits coherent and buildable. Do not commit secrets, `.env` files, databases, uploads or evaluation output; CI rejects tracked runtime data.
- Do not force-push a branch someone else has reviewed unless you tell them.

## 12. Pull requests

1. Make sure the checks in section 5 pass locally.
2. Push your branch and open a pull request against `main`. Fill in the template completely.
3. Describe **what changed and why**, how you tested it, and what you did **not** verify.
4. Call out **behaviour changes**, new configuration, new dependencies and anything that needs a migration or re-index.
5. Link the issue it closes (`Closes #123`).
6. Keep it reviewable. If it grows past a few hundred lines of logic, consider splitting it.
7. Respond to review comments by pushing follow-up commits; the history is squashed on merge.

Draft pull requests are welcome for early feedback.

## 13. Review and merging

- A maintainer reviews every pull request. Expect questions about edge cases, failure modes and security, not only style.
- **All CI jobs must be green** before merging: `test`, the three `draco-core` jobs and `docker-images`.
- Reviewers look for: a test that proves the change, accurate documentation, no widened permissions, no new global state, honest error handling, and no unrelated edits.
- Pull requests are normally **squash-merged** with a clear message. Large multi-part changes may be merged with their commits preserved.
- Maintainers handle releases and version bumps.

## 14. Working on specific areas

**Add a tool.** Use `native_tool` with an explicit input model, permissions, and `requires_approval` / `mutates_state` where relevant; authorize any resource the arguments name. See `examples/d_tool.py` and `examples/g_secure_tenants.py`.

**Add an agent or workload.** Register an `async (ctx, input)` function with `Harness`, route to it with `Router`. See `examples/c_multi_agent.py`.

**Add a plugin.** Declare `provides` / `requires` and use critical hooks for enforcement. See [docs/extension-guide.md](docs/extension-guide.md) and `examples/f_plugin.py`.

**Add a model provider or state store.** Implement the protocol (`ModelProvider`, `ExecutionStore` / `EventStore` / `CheckpointStore`), keep the optional import lazy, add it to the contract tests (the store suite is parametrised across backends), and never include response bodies or credentials in error messages.

**Change retrieval or ingestion in `app/`.** Chunking, embedding or index-schema changes affect existing indexes and may need `scripts/reindex.py`. State this in the pull request, and do not claim a quality improvement without running the evaluation (see [evals.md](evals.md)).

**Change the Docker images or CI.** Build the image locally before pushing, and run any changed workflow logic in the same environment CI uses where you can (for example a Python 3.11 environment for the `draco-core` matrix).

## 15. Getting help

- **Usage questions and ideas:** [GitHub Discussions](https://github.com/AmartyaRanjan/Draco/discussions). See also [SUPPORT.md](SUPPORT.md).
- **Bugs and feature requests:** the issue forms.
- **Security:** [SECURITY.md](SECURITY.md).
- **Where to read first:** [README](README.md), [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md), [docs/api-guide.md](docs/api-guide.md), and the [documentation index](docs/README.md).

## 16. License

By contributing you agree that your contributions are licensed under the [MIT License](LICENSE), the same license as the project.
